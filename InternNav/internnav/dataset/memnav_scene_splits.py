"""Official R2R building splits used by the packed MP3D MemNav dataset.

The source episode JSON files are published by the Matterport3DSimulator R2R
task.  Splitting by building, rather than by generated episode, prevents the
same visual environment from appearing in both training and validation.
"""

R2R_TRAIN_SCENES = frozenset({
    '17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r', '29hnd4uzFmX',
    '2n8kARJN3HM', '5LpN3gDmAk7', '5q7pvUzZiYa', '759xd9YjKW5',
    '7y3sRwLe3Va', '82sE5b5pLXE', '8WUmhLawc2A', 'B6ByNegPMKs',
    'D7G3Y4RVNrH', 'D7N2EKCX4Sj', 'E9uDoFAP3SH', 'EDJbREhghzL',
    'GdvgFV5R1Z5', 'HxpKQynjfin', 'JF19kD82Mey', 'JeFG25nYj2p',
    'JmbYfDe2QKZ', 'PX4nDJXEHrG', 'Pm6F8kyY3z2', 'PuKPg4mmafe',
    'S9hNv5qa7GM', 'SN83YJsR3w2', 'ULsKaCPVFJR', 'Uxmj2M2itWa',
    'V2XKFyX4ASd', 'VFuaQ6m2Qom', 'VLzqgDo317F', 'VVfe2KiqLaN',
    'Vvot9Ly1tCj', 'VzqfbhrpDEA', 'XcA2TqTSSAj', 'YmJkqBEsHnH',
    'ZMojNkEp431', 'aayBHfsNo7d', 'ac26ZMwG7aT', 'b8cTxDM8gDG',
    'cV4RVeZvu5T', 'dhjEzFoUFzH', 'e9zR4mvMWw7', 'gTV8FGcVJC9',
    'gZ6f7yhEvPG', 'i5noydFURQK', 'jh4fc5c5qoQ', 'kEZ7cmS4wCh',
    'mJXqzFtmKg4', 'p5wJjkQkbXX', 'pRbA3pwrgk9', 'qoiz87JEwZ2',
    'r1Q1Z4BcV1o', 'r47D5H71a5s', 'rPc6DW4iMge', 's8pcmisQ38h',
    'sKLMLpTHeUy', 'sT4fr6TAbpF', 'uNb9QFRL6hY', 'ur6pFq6Qu1A',
    'vyrNrziPKCB',
})

R2R_VAL_UNSEEN_SCENES = frozenset({
    '2azQ1b91cZZ', '8194nk5LbLH', 'EU6Fwq7SyZv', 'QUCTc6BB5sX',
    'TbHJrupSAjP', 'X7HyMhZNoso', 'Z6MFQCViBuw', 'oLBMNvg9in8',
    'pLe4wQe7qrG', 'x8F5xyUWy9e', 'zsNo4HB9uLZ',
})

R2R_TEST_SCENES = frozenset({
    '2t7WUuJeko7', '5ZKStnWn8Zo', 'ARNzJeq3xxb', 'RPmz2sHmrrY',
    'UwV83HsGsw3', 'Vt2qJdWjCF2', 'WYY7iVyf5p8', 'YFuZgdQ5vWj',
    'YVUC4YcDtcY', 'fzynW3qQPVF', 'gYvKGZ5eRqb', 'gxdoqLR6rwA',
    'jtcxE69GiFV', 'pa4otMbVnkk', 'q9vSo1VnCiC', 'rqfALeAoiTq',
    'wc2JMjhGNzB', 'yqstnuAEVhm',
})

R2R_SCENE_SPLITS = {
    'r2r_train': R2R_TRAIN_SCENES,
    'r2r_val_unseen': R2R_VAL_UNSEEN_SCENES,
    'r2r_test': R2R_TEST_SCENES,
}

_ALIASES = {
    'all': 'all',
    'train': 'r2r_train',
    'r2r_train': 'r2r_train',
    'val_unseen': 'r2r_val_unseen',
    'r2r_val_unseen': 'r2r_val_unseen',
    'test': 'r2r_test',
    'r2r_test': 'r2r_test',
}


def normalize_scene_split(scene_split):
    """Return the canonical split name and reject accidental typos."""
    name = str(scene_split or 'all').strip().lower().replace('-', '_')
    try:
        return _ALIASES[name]
    except KeyError as exc:
        choices = ', '.join(sorted(_ALIASES))
        raise ValueError(f'unknown scene_split {scene_split!r}; choose one of: {choices}') from exc


def scene_ids_for_split(scene_split):
    """Return the allowed building IDs, or ``None`` for the unfiltered dataset."""
    name = normalize_scene_split(scene_split)
    return None if name == 'all' else R2R_SCENE_SPLITS[name]
