from PIL import Image
from flask import Flask, request, jsonify
from policy_agent import NavDP_Agent
from deterministic_seed import apply_seed
import numpy as np
import cv2
import imageio
import time
import datetime
import hashlib
import json
import os
from PIL import Image, ImageDraw, ImageFont
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8888)
parser.add_argument("--checkpoint",type=str,default="/home/PJLAB/caiwenzhe/Desktop/navdp_bench/baselines/navdp/checkpoints/cross-waic-final4-125.ckpt")
args = parser.parse_known_args()[0]

app = Flask(__name__)
navdp_navigator = None
navdp_fps_writer = None


def memory_queue_fingerprints(navigator):
    """Content fingerprints for fail-closed read-only resample contracts."""
    fingerprints = []
    for queue in navigator.memory_queue:
        digest = hashlib.sha256()
        digest.update(str(len(queue)).encode())
        for item in queue:
            array = np.ascontiguousarray(np.asarray(item))
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        fingerprints.append(digest.hexdigest())
    return fingerprints


def _form_boolean(name, default=False):
    value = request.form.get(name)
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean form value for {name}: {value}")


def _score_timesteps_from_form():
    raw = request.form.get("score_timesteps")
    if raw is None or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(parsed, list):
        raise ValueError("score_timesteps must be a JSON list or CSV sequence")
    return [int(value) for value in parsed]


def _critic_receipt(values):
    scores = np.asarray(values, dtype=np.float64)
    critic_max = float(scores.max())
    return {
        "critic_max": critic_max,
        "critic_min": float(scores.min()),
        "critic_threshold": float(navdp_navigator.stop_threshold),
        "critic_fallback_applied": bool(
            critic_max < float(navdp_navigator.stop_threshold)),
    }


@app.route("/navigator_reset",methods=['POST'])
def navdp_reset():
    global navdp_navigator,navdp_fps_writer
    payload = request.get_json()
    intrinsic = np.asarray(payload.get('intrinsic'), dtype=np.float64)
    threshold = float(payload.get('stop_threshold'))
    batchsize = int(payload.get('batch_size'))
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        return jsonify({"error": "intrinsic must be a finite 3x3 matrix"}), 400
    if not np.isfinite(threshold):
        return jsonify({"error": "stop_threshold must be finite"}), 400
    if batchsize < 1:
        return jsonify({"error": "batch_size must be positive"}), 400
    seed = payload.get('seed')
    apply_seed(seed)
    if navdp_navigator is None:
        navdp_navigator = NavDP_Agent(intrinsic,
                                image_size=224,
                                memory_size=8,
                                predict_size=24,
                                temporal_depth=16,
                                heads=8,
                                token_dim=384,
                                navi_model=args.checkpoint,
                                device='cuda:0')
        navdp_navigator.reset(batchsize,threshold)
    else:
        navdp_navigator.reset(batchsize,threshold)

    if os.environ.get("NAVDP_DISABLE_VIDEO", "0") == "1":
        navdp_fps_writer = None
    elif navdp_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    else:
        navdp_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    return jsonify({
        "algo": "navdp",
        "stop_threshold": threshold,
        "threshold_semantics": "critic_score_fallback",
        "batch_size": batchsize,
        "checkpoint_contract": navdp_navigator.checkpoint_contract,
    })

@app.route("/navigator_reset_env",methods=['POST'])
def navdp_reset_env():
    global navdp_navigator
    navdp_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo":"navdp"})


@app.route("/memory_replay_step", methods=['POST'])
def navdp_memory_replay_step():
    """Append one frozen decision image without running diffusion."""
    global navdp_navigator
    if navdp_navigator is None:
        return jsonify({"error": "navigator is not initialized"}), 409
    image_file = request.files['image']
    batch_size = navdp_navigator.batch_size
    image = Image.open(image_file.stream).convert('RGB')
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    queue_lengths = navdp_navigator.append_observation(image)
    return jsonify({
        "algo": "navdp",
        "queue_lengths": queue_lengths,
        "memory_size": int(navdp_navigator.memory_size),
        "diffusion_sampled": False,
    })

@app.route("/pointgoal_step",methods=['POST'])
def navdp_step_xy():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.zeros_like(goal_x)),axis=1)
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_pointgoal(goal,image,depth)
    phase3_time = time.time()
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))

    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
        'diffusion_seed': diffusion_seed,
        **_critic_receipt(all_values),
    })


@app.route("/pixelgoal_step",methods=['POST'])
def navdp_step_pixel():
    global navdp_navigator,navdp_fps_writer
    
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y),axis=1)
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_pixelgoal(goal,image,depth)
    phase3_time = time.time()
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist(),
                    'diffusion_seed': diffusion_seed})

@app.route("/imagegoal_step",methods=['POST'])
def navdp_step_image():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_file = request.files['goal']
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("image.jpg",image)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    goal = Image.open(goal_file.stream)
    goal = goal.convert('RGB')
    goal = np.asarray(goal)
    goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
    cv2.imwrite("goal.jpg",goal)
    goal = goal.reshape((batch_size, -1, goal.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_imagegoal(goal,image,depth)
    phase3_time = time.time()
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
        'diffusion_seed': diffusion_seed,
        **_critic_receipt(all_values),
    })


@app.route("/imagegoal_resample", methods=["POST"])
def navdp_resample_imagegoal():
    """Resample candidates from the current FIFO without appending an image."""
    global navdp_navigator
    if navdp_navigator is None:
        return jsonify({"error": "navigator is not initialized"}), 409
    batch_size = navdp_navigator.batch_size

    image = Image.open(request.files['image'].stream).convert('RGB')
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    goal = Image.open(request.files['goal'].stream).convert('RGB')
    goal = cv2.cvtColor(np.asarray(goal), cv2.COLOR_RGB2BGR)
    goal = goal.reshape((batch_size, -1, goal.shape[1], 3))

    depth = Image.open(request.files['depth'].stream).convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    before_lengths = [len(queue) for queue in navdp_navigator.memory_queue]
    before_hashes = memory_queue_fingerprints(navdp_navigator)
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, _trajectory_mask = (
        navdp_navigator.resample_imagegoal(goal, image, depth))
    after_lengths = [len(queue) for queue in navdp_navigator.memory_queue]
    after_hashes = memory_queue_fingerprints(navdp_navigator)
    if after_lengths != before_lengths or after_hashes != before_hashes:
        return jsonify({"error": "resampling mutated NavDP memory"}), 500
    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
        'diffusion_seed': diffusion_seed,
        **_critic_receipt(all_values),
        'memory_mutated': False,
        'queue_lengths': after_lengths,
        'queue_hashes_before': before_hashes,
        'queue_hashes_after': after_hashes,
    })

@app.route("/nogoal_step",methods=['POST'])
def navdp_step_nogoal():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_nogoal(image,depth)
    phase3_time = time.time()
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist(),
                    'diffusion_seed': diffusion_seed})

@app.route("/navdp_step_ip_mixgoal",methods=['POST'])
def navdp_step_ip_mixgoal():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    batch_size = navdp_navigator.batch_size
    
    point_goal_data = json.loads(request.form.get('goal_data'))
    point_goal_x = np.array(point_goal_data['goal_x'])
    point_goal_y = np.array(point_goal_data['goal_y'])
    point_goal = np.stack((point_goal_x,point_goal_y,np.zeros_like(point_goal_x)),axis=1)
    
    image_goal_file = request.files['image_goal']
    image_goal = Image.open(image_goal_file.stream)
    image_goal = image_goal.convert('RGB')
    image_goal = np.asarray(image_goal)
    image_goal = cv2.cvtColor(image_goal, cv2.COLOR_RGB2BGR)
    cv2.imwrite("goal.jpg",image_goal)
    image_goal = image_goal.reshape((batch_size, -1, image_goal.shape[1], 3))
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_point_image_goal(point_goal,image_goal,image,depth)
    phase3_time = time.time()
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
        'diffusion_seed': diffusion_seed,
        **_critic_receipt(all_values),
    })


@app.route("/mixgoal_resample", methods=["POST"])
def navdp_resample_mixgoal():
    """Read-only mixed image/point proposal from the current NavDP FIFO."""
    global navdp_navigator
    if navdp_navigator is None:
        return jsonify({"error": "navigator is not initialized"}), 409
    batch_size = navdp_navigator.batch_size
    point_goal_data = json.loads(request.form.get('goal_data'))
    point_goal_x = np.asarray(point_goal_data['goal_x'])
    point_goal_y = np.asarray(point_goal_data['goal_y'])
    point_goal = np.stack(
        (point_goal_x, point_goal_y, np.zeros_like(point_goal_x)), axis=1)

    image = Image.open(request.files['image'].stream).convert('RGB')
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    image_goal = Image.open(request.files['image_goal'].stream).convert('RGB')
    image_goal = cv2.cvtColor(np.asarray(image_goal), cv2.COLOR_RGB2BGR)
    image_goal = image_goal.reshape(
        (batch_size, -1, image_goal.shape[1], 3))
    control_goal = None
    if 'control_goal' in request.files:
        control_goal = Image.open(
            request.files['control_goal'].stream).convert('RGB')
        control_goal = cv2.cvtColor(
            np.asarray(control_goal), cv2.COLOR_RGB2BGR)
        control_goal = control_goal.reshape(
            (batch_size, -1, control_goal.shape[1], 3))
    depth = Image.open(request.files['depth'].stream).convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis].astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    try:
        score_goal_contrast = _form_boolean("score_goal_contrast", False)
        score_timesteps = _score_timesteps_from_form()
        score_noise_samples = int(
            request.form.get("score_noise_samples", 1))
        score_seed = int(request.form.get(
            "score_seed", request.form.get('diffusion_seed', 0)))
        if score_noise_samples < 1 or score_noise_samples > 16:
            raise ValueError("score_noise_samples must lie in [1, 16]")
        if control_goal is not None and not score_goal_contrast:
            raise ValueError(
                "control_goal requires score_goal_contrast=true")
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    before_lengths = [len(queue) for queue in navdp_navigator.memory_queue]
    before_hashes = memory_queue_fingerprints(navdp_navigator)
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, _trajectory_mask = (
        navdp_navigator.resample_point_image_goal(
            point_goal, image_goal, image, depth))
    goal_contrast = None
    if score_goal_contrast:
        try:
            goal_contrast = navdp_navigator.score_imagegoal_trajectories(
                image_goal,
                image,
                depth,
                all_trajectory,
                control_imagegoal=control_goal,
                timesteps=score_timesteps,
                noise_samples=score_noise_samples,
                seed=score_seed,
            )
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
    after_lengths = [len(queue) for queue in navdp_navigator.memory_queue]
    after_hashes = memory_queue_fingerprints(navdp_navigator)
    if after_lengths != before_lengths or after_hashes != before_hashes:
        return jsonify({"error": "mixed resampling mutated NavDP memory"}), 500
    payload = {
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
        'diffusion_seed': diffusion_seed,
        **_critic_receipt(all_values),
        'memory_mutated': False,
        'queue_lengths': after_lengths,
        'queue_hashes_before': before_hashes,
        'queue_hashes_after': after_hashes,
    }
    if goal_contrast is not None:
        payload['goal_contrast'] = goal_contrast
    return jsonify(payload)
    

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=args.port)
