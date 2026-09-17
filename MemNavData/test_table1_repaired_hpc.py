import json
from pathlib import Path
import pytest

from MemNavData.summarize_table1_repaired import statistics, summarize


def test_statistics_are_paired_and_role_local():
    rows=[]
    for s, n, g, take in [("a",0,1,True),("b",1,1,False),("c",1,0,True),("d",0,1,True)]:
        rows.append(dict(scene=s,native=dict(reached=n,spl=.5*n),cec=dict(reached=g,spl=.4*g),
            pair=dict(cec_takeover=take,no_takeover_exact_native=not take)))
    v=statistics(rows)
    assert v["gain"]==2 and v["loss"]==1 and v["risk_difference"]==.25
    assert v["arms"]["native"]["successes"]==2 and v["arms"]["cec"]["successes"]==3
    assert v["exact_mcnemar_p"]==1 and v["exact_native_queries"]==1


def test_missing_task_does_not_become_failure_sr(tmp_path):
    plan=tmp_path/"plan.json"; out=tmp_path/"summary.json"
    plan.write_text(json.dumps(dict(cells=[dict(index=0)])))
    with pytest.raises(RuntimeError,match="Incomplete population"):
        summarize(plan,tmp_path,out)
    v=json.loads(out.read_text())
    assert v["complete"] is False and "groups" not in v and v["verified_cells"]==0


def test_fixed_arrays_cover_all_cells_once():
    gates={0,28,56,84,126,168}
    remaining={i for lo,hi in [(1,27),(29,55),(57,83),(85,125),(127,167),(169,209)] for i in range(lo,hi+1)}
    assert len(gates)==6 and len(remaining)==204 and not (gates&remaining)
    assert gates|remaining==set(range(210))
    text=Path(__file__).with_name("submit_table1_repaired_hpc.sh").read_text()
    assert "--array=0,28,56,84,126,168%4" in text
    assert "--array=1-27,29-55,57-83,85-125,127-167,169-209%4" in text
    assert '--dependency="afterok:${TABLE1_GATE}"' in text
