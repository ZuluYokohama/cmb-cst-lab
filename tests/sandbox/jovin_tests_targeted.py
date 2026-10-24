import numpy as np
from solution import six_responses, fit_targeted, score_targeted, choose_blind_spots, ensure_budget, wilson_interval


def toy_training():
    rng = np.random.default_rng(668923)
    b,r = rng.normal(size=(160,17)),rng.normal(size=(160,6))
    labels = np.array(['null']*100+['quadratic']*20+['compact']*20+['filament']*20)
    for i in range(3):
        b[100+20*i:120+20*i,i] += .5
        r[100+20*i:120+20*i,i] += 2
    return b,r,labels


def test_six_fixed_responses_have_semantic_indices():
    x = np.arange(90.)[None]
    # orientation raw/rank Euler(level0), smoothing120 raw/rank fourth/Euler.
    assert np.array_equal(six_responses(x),[[7,16,73,79,82,88]])


def test_model_is_not_mutated_by_scoring_and_response_signal_is_visible():
    b,r,labels = toy_training()
    model = fit_targeted(b,r,labels)
    before = str(model)
    s = score_targeted(b[:5],r[:5],model)
    changed = score_targeted(b[:5],r[:5]+5,model)
    assert str(model)==before and np.array_equal(s[:,:2],changed[:,:2])
    assert np.all(changed[:,2]>s[:,2])
    assert model['training_counts']=={'null':100,'quadratic':20,'compact':20,'filament':20}


def test_selection_requires_blind_spot_and_uses_fixed_ties():
    rows = [dict(id='a',family='compact',baseline_rate=.16,gain=.9,amplitude=.1),
        dict(id='b',family='compact',baseline_rate=.1,gain=.2,amplitude=.2),
        dict(id='c',family='compact',baseline_rate=.1,gain=.2,amplitude=.1)]
    assert choose_blind_spots(rows)=={'quadratic':None,'compact':'c','filament':None}


def test_budget_and_invalid_inputs_stop():
    ensure_budget(1799.9)
    for value in (1800.,np.inf,np.nan):
        try:
            ensure_budget(value)
        except TimeoutError:
            continue
        raise AssertionError('budget escaped')
    try:
        six_responses(np.zeros((2,89)))
    except ValueError:
        pass
    else:
        raise AssertionError('invalid features accepted')


def test_wilson_extremes_and_actual_cuda_scores():
    assert wilson_interval([False]*100)[1]<.075
    assert wilson_interval([True]*100)[0]>.9
    b,r,labels = toy_training()
    scores = score_targeted(b[:8],r[:8],fit_targeted(b,r,labels))
    assert np.allclose(scores,GPU_ATTESTATION,rtol=1e-11,atol=1e-11)


GPU_ATTESTATION = [[0.2690680834940743, 1.613170134682976, 0.3409646800263781], [-0.4722693050555292, 1.6147594827075031, 0.19835339080017717], [0.20261577587087343, 1.0181561267597732, 0.9310517867558045], [-0.0605573486952266, -0.05073803869497162, 1.7777925646770951], [-0.3191329903327726, -0.9447025039045691, 0.9807519733122333], [0.5078761696162685, 1.3803982188976887, 1.6546961351933325], [-0.5119921395811813, -0.9638474933615729, -1.0851012940747953], [-1.344401587877375, -0.4069202685139969, 1.2159936040327144]]
