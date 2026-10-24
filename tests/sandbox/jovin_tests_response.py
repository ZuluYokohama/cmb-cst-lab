import numpy as np
from solution import (spherical_patch_locations, excursion_euler_reference,
    extract_response_reference, fit_response_model, score_response_reference,
    upper_tail_pvalues, paired_win_pvalue, foreground_proxy, localized_template)


def test_rotated_spherical_grids_preserve_center_distance_and_quarter_turn():
    loc = spherical_patch_locations(.8, .7, [0.,np.pi/2], size=19)
    vector = np.stack([np.sin(loc[...,0])*np.cos(loc[...,1]), np.sin(loc[...,0])*np.sin(loc[...,1]), np.cos(loc[...,0])], axis=-1)
    assert np.allclose(np.linalg.norm(vector, axis=-1), 1., atol=1e-14)
    assert np.allclose(loc[0,9,9], [.8,.7], atol=1e-14)
    north = np.array([-np.cos(.8)*np.cos(.7),-np.cos(.8)*np.sin(.7),np.sin(.8)])
    assert vector[1,9,-1]@north > 0
    assert np.allclose(vector[1], np.rot90(vector[0], 1), atol=1e-13)


def test_euler_characteristic_matches_independent_graph_and_cycle_counts():
    rng = np.random.default_rng(771233)
    x = rng.normal(size=(20,21))
    observed = excursion_euler_reference(x)[0]
    z = (x-x.mean())/x.std()
    for i,threshold in enumerate((-1.,0.,1.)):
        a = z > threshold
        seen, components = set(), 0
        for r,c in np.argwhere(a):
            if (int(r),int(c)) in seen:
                continue
            components += 1
            pending = [(int(r),int(c))]
            while pending:
                v = pending.pop()
                if v in seen:
                    continue
                seen.add(v)
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr,nc = v[0]+dr,v[1]+dc
                    if 0 <= nr < len(a) and 0 <= nc < a.shape[1] and a[nr,nc] and (nr,nc) not in seen:
                        pending.append((nr,nc))
        background_seen, holes = set(), 0
        for r,c in np.argwhere(~a):
            if (int(r),int(c)) in background_seen:
                continue
            pending, boundary = [(int(r),int(c))], False
            while pending:
                v = pending.pop()
                if v in background_seen:
                    continue
                background_seen.add(v)
                boundary |= v[0] in (0,len(a)-1) or v[1] in (0,a.shape[1]-1)
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        nr,nc = v[0]+dr,v[1]+dc
                        if 0<=nr<len(a) and 0<=nc<a.shape[1] and not a[nr,nc] and (nr,nc) not in background_seen:
                            pending.append((nr,nc))
            holes += not boundary
        assert observed[i] == (components-holes)/x.size


def test_repeated_views_have_zero_responses_and_correct_dimensions():
    x = np.random.default_rng(452891).normal(size=(2,32,33))
    fields = np.broadcast_to(x[:,None,None,None], (2,3,4,2,32,33)).copy()
    baseline,response = extract_response_reference(fields)
    assert baseline.shape == (2,17) and response.shape == (2,90)
    assert np.allclose(response, 0., atol=1e-13)


def test_ridge_model_is_training_only_and_scores_extra_geometry():
    rng = np.random.default_rng(885101)
    b = rng.normal(size=(80,3))
    r = b@rng.normal(size=(3,4))+.1*rng.normal(size=(80,4))
    model = fit_response_model(b,r)
    assert model['training_draws'] == 80 and max(model['condition_numbers']) < 1e5
    before = str(model)
    s = score_response_reference(b[:2],r[:2],model)
    changed = score_response_reference(b[:2],r[:2]+20,model)
    assert str(model) == before and np.array_equal(s[:,0],changed[:,0])
    assert np.all(changed[:,1] > s[:,1])


def test_upper_tail_ties_plus_one_and_exact_paired_test():
    p = upper_tail_pvalues([[0.,1.],[3.,3.]], [[1.,1.],[1.,2.],[2.,3.]])
    assert np.array_equal(p, [[1.,1.],[.25,.5]])
    assert paired_win_pvalue([False]*5,[True]*5) == (5,0,1/32)
    assert paired_win_pvalue([True]*5,[False]*5) == (0,5,1.)
    assert paired_win_pvalue([True,False],[True,False]) == (0,0,1.)


def test_invalid_feature_and_score_inputs_fail_closed():
    bad = (np.zeros((1,3,4,2,16,16)), np.full((1,3,4,2,18,18),np.nan), np.ones((1,3,4,2,18,18)))
    for fields in bad:
        try:
            extract_response_reference(fields)
        except ValueError:
            continue
        raise AssertionError('invalid field accepted')
    try:
        upper_tail_pvalues([[np.nan]],[[1.]])
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite score accepted')


def test_foreground_and_localized_templates_are_finite_and_local():
    theta,phi = np.meshgrid(np.linspace(.01,np.pi-.01,30),np.linspace(0,2*np.pi,40),indexing='ij')
    assert np.all(np.isfinite(foreground_proxy(theta,phi)))
    for kind in ('compact','filament'):
        field = localized_template(theta,phi,(.8,.7),kind,np.random.default_rng(123))
        assert np.isfinite(field).all() and np.any(field != 0) and np.any(field == 0)


def test_actual_cuda_response_fixture_matches_shipped_cpu_code():
    fixture = GPU_ATTESTATION
    fields = np.random.default_rng(fixture['seed']).normal(size=(2,3,4,2,32,33))
    b,r = extract_response_reference(fields)
    assert np.allclose(b,fixture['baseline'],rtol=1e-11,atol=1e-11)
    assert np.allclose(r,fixture['response'],rtol=1e-11,atol=1e-11)


GPU_ATTESTATION = {'seed': 551940, 'baseline': [[0.0052042792389378, -0.02103179260283741, -0.143514842420307, 0.6863619844665891, 0.6778644261991899, 0.6939531336448818, 0.6795821335183354, 0.7074230493746693, 0.0063247515628487155, -0.05932306840208268, -0.7379300723719421, -0.004619240733020523, -0.013554922649676962, 0.016130235278388774, -0.06439393939393939, 0.08049242424242424, 0.10984848484848485], [0.04986641719683872, 0.19184597759443603, 0.09764053533032202, 0.6723275624726651, 0.6833121577127885, 0.7086769522932571, 0.7037032560028655, 0.7195570709549459, 0.01656936917233956, -0.0059866234930916916, 0.20053186059115974, -0.01429813268617558, -0.029056648769818717, -0.017417143164898584, -0.05776515151515152, 0.054924242424242424, 0.10795454545454546]], 'response': [[0.0029257775826970426, 0.054031380125835325, 0.5377397374881279, 0.0006247116160213572, -0.004104706016585288, -0.01611285666551127, 0.004734848484848481, -0.0028409090909090884, 0.005681818181818163, 0.009086033054076372, 0.14779941561719667, 1.998731285921133, 0.02366594805403858, 0.06796463967325131, 0.002789140148279, 0.0037878787878787984, 0.00031565656565656075, -0.0056818181818182045, 0.01333065149427459, 0.05232394632305437, 1.8031149029910218, 0.01681148368704568, 0.08560630924341953, 0.013401886770004657, -0.00662878787878788, -0.005366161616161616, 0.003156565656565649, -0.0014271901055574639, 0.04620696430429598, 0.21039156942546544, -0.02007924843989624, -0.1017851175234061, -0.0038825400086600816, -0.008207070707070704, -0.012310606060606064, 0.00031565656565656075, 0.01629573642387038, 0.25126674348973493, 3.2267168949141913, 0.056556599480015374, 0.16313057042002654, 0.012157689829367789, -0.0009469696969696961, 0.009785353535353536, -1.3877787807814457e-17, -0.00020836347827431814, 0.06623082472322642, 1.2083250402481027, -0.022602911766140488, -0.10004649748441381, -0.013383027276683591, -0.003472222222222217, -0.007575757575757583, -0.007575757575757597, -0.010447945772586077, 0.09567553384001233, 0.8403678060593478, 0.01554670768316624, 0.03573120082942592, -0.019194684222812026, 0.01136363636363636, 0.010416666666666671, -0.0009469696969696961, 0.003501666315759853, 0.1497517460164257, 2.4451472944656825, 0.02966191883339929, 0.11035460347823103, -0.0025111366161383085, 0.009469696969696975, 0.005681818181818177, -0.0037878787878787984, -0.008139056590876442, -0.121355884669366, -1.9361596733286626, -0.02518018038016692, -0.07624767365365775, -0.029912816331294594, 0.0, -0.014204545454545456, 0.0, 0.011845422271430756, 0.1218824658900263, 1.4147241005732658, 0.04024107641124989, 0.11886175030428034, 0.006772750327955669, 0.010416666666666671, 0.0018939393939393923, 0.0], [-0.012983195141935263, -0.10696640191060444, -1.7188494786382933, 0.016423567316283083, 0.0436521266801256, 0.031850380911605976, -0.0037878787878787845, 0.021148989898989896, 0.005366161616161616, -0.00204099750847471, -0.20384167427053612, -2.1942605355604297, 0.0344294855161202, 0.08654754007111837, -0.023871016718864876, -0.0037878787878787845, 0.018623737373737376, 0.011679292929292928, -0.0021821216145177912, 0.07920865556129092, 0.8890834377561598, -0.017015688323784854, -0.035392508624008015, -0.008532359204787322, 0.0015782828282828315, -0.011679292929292928, 0.01672979797979797, 0.016061764853461655, -0.02788436527909295, -0.5269198823697523, -0.004088082586638224, -0.012264786872053894, 0.004953994066775283, 0.00031565656565656075, -0.009785353535353522, -0.006944444444444461, 0.012039208801913445, 0.05255217497986005, 0.7118463999708299, -0.002318035624381742, 0.03302821072705698, -0.025495765111566863, 0.0091540404040404, -0.016098484848484862, 0.0025252525252525276, -0.0013477888250487315, -0.05488174094777047, -1.4706736045420603, 0.015280365389228607, 0.038914253394064346, -0.0029625810429273814, 0.00220959595959596, 0.010732323232323232, 0.005366161616161616, -0.008794339429213816, -0.06444063947050918, -1.2450213901302933, 0.027860526454287438, 0.05271588638344159, 0.0219273058492592, -0.0028409090909090884, 0.025568181818181816, -0.008522727272727265, -0.010340865557571006, -0.024924727631553262, 0.8997570861793216, 0.032356989021268476, 0.07838329052063883, -0.016378659215720556, 0.0037878787878787915, 0.016098484848484848, 0.010416666666666671, -0.024803883139241478, -0.030798545391726242, -0.7175021069334158, 0.025972844930384213, 0.023771630753493134, 0.03520538413369702, -0.010416666666666657, 0.03598484848484849, -0.0009469696969696961, 0.004612021705673239, -0.07330904821248599, 0.15342722752294402, 0.0089037971025453, 0.03118133433446644, -0.01923818069563519, -0.0028409090909090884, 0.005681818181818191, -0.0028409090909090884]]}
