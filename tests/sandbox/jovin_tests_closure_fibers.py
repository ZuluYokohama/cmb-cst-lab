import numpy as np
from solution import features_numpy, ratios_numpy, moments_with_derivatives, projection_indices, robust_scales, quantize
from solution import harmonic_layout,build_planes,apply_angles,invariants,tangent_reference,matching_bins,initial_kicks
from solution import bin_separation

GPU_ATTESTATION = {'seed': 929173, 'devices': [{'device': 'cuda:0', 'name': 'NVIDIA GeForce RTX 5070', 'features': [18.299722685024715, 12.423572548743259, 2.7206157768332617, 9.041131696858624, 6.320515920025362, 0.18043538381383972, -0.000542601393364704, 0.24874656716741703, 23.980764532074218, 14.386699209493663, 1.150546465508858, 10.1671391681464, 9.016592702637542, 0.061942976220244826, -0.14268575290724347, 0.2616854427940167, 18.34612003191868, 14.10351816177748, 1.2019819745063804, 8.71340053567723, 7.511418561170849, 0.07472434354757736, -0.2959290311709337, 0.25433133438667144, 21.948306450575625, 12.827070284040587, -1.04764822515729, 8.170020071075408, 9.217668296232699, -0.06243832395738564, -0.4777131880013382, 0.41005072425282685, 5.82978401095665, 2.2174612515033045, 0.4774310404671165, 1.9602629191628564, 1.5685178701328595, -0.2595013538823301, 2.424039065623226, 1.0346704572197338, 0.4498110724484385, 1.5148985050306623, 1.76476190363873, 0.5943915392596827, -6.3504341590570395, -6.554065898911469, -4.833464384945426, -4.676997539776845, 0.04482272124851601, 0.012999479044128844, -0.03494268005739853, 0.1285617146068471], 'ratios': [0.004551203555625844, 0.007663477587998875], 'jacobian': [[-0.3070844331704403, 0.04280997951690194, -0.16612077942641656, 0.47431237238165636], [0.12395214079574696, 0.016529374425462616, 0.03516874957260773, -0.17996308510338932], [-0.09156614618734668, 0.02966967697118228, -0.06547601492690441, 0.14717464363913352], [-0.09156614618734668, 0.02966967697118228, -0.06547601492690441, 0.14717464363913352], [0.0, 0.0, 0.0, 0.0], [-0.005458996670163608, 0.0016366515805304235, -0.0037788820705243684, 0.00872934381808356], [-0.012334539276027305, -0.004664609039824702, 0.02642226799182098, -0.014768426843508238], [0.012487398131997599, -0.0045463194633963465, 0.012288565068427333, -0.014901305620530436], [-0.23164680170263574, 0.22142236544490718, -0.2509947942068412, 0.024497485705237], [0.255565518200273, -0.16107835964903874, 0.008099205527194954, -0.14352262846026345], [0.011959358248818636, 0.030172002897934216, -0.12144779433982311, -0.05951257137751323], [0.011959358248818636, 0.030172002897934216, -0.12144779433982311, -0.05951257137751323], [0.0, 0.0, 0.0, 0.0], [0.0003928635363331017, 0.001685193743298398, -0.006231763730418517, -0.0029266956500140465], [0.0017852463617589799, -0.003418757406618614, -0.010194241928009088, -0.0015708000595660615], [0.001494760684610054, 0.0009216126479193924, -0.010230240354405402, 0.006252308146584475], [-0.026202641996337983, -0.09568705764823746, 0.358557196124581, -0.08716475417277886], [0.38182623918529135, -0.2756440201638424, 0.14248996554760845, -0.4863714457745042], [0.17781179859447668, -0.18566553890603993, 0.2505235808360947, -0.2867680999736415], [0.17781179859447668, -0.18566553890603993, 0.2505235808360947, -0.2867680999736415], [0.0, 0.0, 0.0, 0.0], [0.010095985641943481, -0.010617294926805774, 0.014466768427047824, -0.016361708945943226], [-0.007825827945376647, -0.005717780140630532, 0.00015869313703707503, 0.008965202614042676], [0.02730325564595997, -0.00954975801368325, -0.007621530824522067, -0.009714306243570165], [0.13587429639537185, 0.5685140130885373, 0.08979720019932264, -0.07630942507381482], [0.3892622452975805, -0.22427190413800455, -0.08381111499914277, -0.3826366531280204], [0.26256827084647616, 0.17212105447526638, 0.0029930426000899304, -0.22947303910091763], [0.26256827084647616, 0.17212105447526638, 0.0029930426000899304, -0.22947303910091763], [0.0, 0.0, 0.0, 0.0], [0.016789363178363197, 0.010520974106525695, 0.0001021244529795188, -0.014716084602764091], [0.009853047648492225, -0.030881067384675158, 0.0041372546028454055, -0.012339205398363702], [0.009200932771667912, -0.007509511821382342, 0.012474783432306352, -0.01952327898718104], [-0.07598535674174661, -0.0, 0.0, -0.0], [0.17227890532278398, -0.0, 0.0, -0.0], [0.04814677429051867, -0.0, 0.0, -0.0], [0.0, 0.15695680524645847, 0.0, -0.0], [0.0, -0.08077576665905438, 0.0, -0.0], [0.0, 0.03809051929370203, 0.0, -0.0], [0.0, 0.0, -0.049517200018337475, 0.0], [0.0, -0.0, -0.04998638510222169, 0.0], [0.0, 0.0, -0.0497517925602796, 0.0], [-0.0, -0.0, 0.0, -0.07706820167421372], [-0.0, 0.0, 0.0, -0.09346895962372273], [-0.0, 0.0, 0.0, -0.08526858064896825], [0.0468923641171301, -5.39329830255824e-05, 0.06309520895098028, -0.09726350234072303], [0.0042684976433691926, -0.027255614888658852, 0.10685021623650875, 0.09627472995081418], [-0.06274424281506633, 0.04906684091267763, -0.1423179282672935, 0.21242406658703503], [-0.04070569660679274, -0.11579104482091146, -0.014371045508180953, 0.2134713903143579], [-0.010654795309814485, 0.004212863334213924, 0.0007137447344568836, -0.0015943608866439787], [0.0018294577439231705, 0.0022439310916146996, 0.0003455212685148898, 0.004299567092866638], [-0.0001435041117039985, 0.00011108420211753702, 0.0012750500570991841, 0.0037288521902763645], [-0.00042644381386111683, -0.01897076615565782, -0.004414438765750729, -0.0018197491322891925]], 'errors': {'features': 6.302339632288665e-16, 'ratios': 8.673617379884035e-19, 'jacobian_five_point': 4.499468482902334e-12, 'optimized_jacobian_autograd': 2.61097063226752e-16, 'optimized_target_value': 8.44206465503319e-17, 'optimized_target_gradient': 2.3353602478736e-17, 'analytic_response': 7.137645971266811e-17}, 'target_value': -0.3151084124465189, 'target_gradient': [-0.005348534053961789, -0.005477489513392505, -0.0004690641769977328, 0.010984313172502022], 'nonclosed_control_theta': [0.0, 0.7853981633974483], 'identity_control_gap': [0.0, 0.0], 'solver_control_seconds': 8.112733400000002}, {'device': 'cuda:1', 'name': 'NVIDIA GeForce GTX 1660 Ti', 'features': [18.299722685024715, 12.423572548743259, 2.7206157768332617, 9.041131696858624, 6.320515920025362, 0.18043538381383972, -0.000542601393364954, 0.24874656716741703, 23.980764532074218, 14.386699209493663, 1.1505464655088564, 10.1671391681464, 9.016592702637544, 0.06194297622024474, -0.14268575290724347, 0.2616854427940167, 18.34612003191868, 14.10351816177748, 1.2019819745063804, 8.71340053567723, 7.51141856117085, 0.07472434354757736, -0.2959290311709337, 0.25433133438667144, 21.948306450575625, 12.827070284040587, -1.04764822515729, 8.170020071075408, 9.217668296232699, -0.06243832395738564, -0.4777131880013383, 0.41005072425282685, 5.82978401095665, 2.2174612515033045, 0.4774310404671165, 1.9602629191628564, 1.5685178701328595, -0.2595013538823301, 2.424039065623226, 1.0346704572197338, 0.4498110724484385, 1.5148985050306623, 1.76476190363873, 0.5943915392596827, -6.3504341590570395, -6.554065898911469, -4.833464384945426, -4.676997539776845, 0.044822721248515975, 0.012999479044128844, -0.03494268005739853, 0.1285617146068468], 'ratios': [0.004551203555625843, 0.007663477587998875], 'jacobian': [[-0.3070844331704404, 0.04280997951690195, -0.16612077942641656, 0.47431237238165636], [0.1239521407957469, 0.01652937442546261, 0.03516874957260774, -0.17996308510338926], [-0.09156614618734674, 0.02966967697118228, -0.06547601492690441, 0.14717464363913355], [-0.09156614618734674, 0.02966967697118228, -0.06547601492690441, 0.14717464363913355], [0.0, 0.0, 0.0, 0.0], [-0.00545899667016361, 0.0016366515805304235, -0.0037788820705243684, 0.00872934381808356], [-0.0123345392760273, -0.004664609039824702, 0.02642226799182098, -0.01476842684350825], [0.012487398131997608, -0.004546319463396345, 0.012288565068427333, -0.014901305620530434], [-0.23164680170263577, 0.22142236544490723, -0.25099479420684123, 0.024497485705236972], [0.255565518200273, -0.16107835964903874, 0.008099205527194956, -0.14352262846026342], [0.011959358248818622, 0.030172002897934244, -0.12144779433982314, -0.05951257137751323], [0.011959358248818622, 0.030172002897934244, -0.12144779433982314, -0.05951257137751323], [0.0, 0.0, 0.0, 0.0], [0.0003928635363331014, 0.0016851937432983992, -0.006231763730418519, -0.002926695650014047], [0.0017852463617589818, -0.003418757406618611, -0.010194241928009088, -0.0015708000595660667], [0.0014947606846100543, 0.0009216126479193876, -0.010230240354405408, 0.006252308146584477], [-0.026202641996337924, -0.09568705764823747, 0.358557196124581, -0.08716475417277886], [0.3818262391852914, -0.2756440201638424, 0.14248996554760843, -0.4863714457745043], [0.17781179859447674, -0.18566553890603993, 0.2505235808360947, -0.2867680999736416], [0.17781179859447674, -0.18566553890603993, 0.2505235808360947, -0.2867680999736416], [0.0, 0.0, 0.0, 0.0], [0.010095985641943485, -0.010617294926805774, 0.014466768427047824, -0.016361708945943233], [-0.007825827945376652, -0.005717780140630525, 0.00015869313703707158, 0.008965202614042682], [0.02730325564595997, -0.009549758013683247, -0.007621530824522067, -0.009714306243570161], [0.13587429639537182, 0.5685140130885372, 0.08979720019932265, -0.07630942507381484], [0.38926224529758047, -0.22427190413800457, -0.08381111499914279, -0.3826366531280205], [0.26256827084647616, 0.17212105447526632, 0.0029930426000899304, -0.22947303910091765], [0.26256827084647616, 0.17212105447526632, 0.0029930426000899304, -0.22947303910091765], [0.0, 0.0, 0.0, 0.0], [0.016789363178363197, 0.010520974106525692, 0.0001021244529795188, -0.014716084602764093], [0.009853047648492234, -0.030881067384675158, 0.004137254602845401, -0.012339205398363699], [0.009200932771667917, -0.007509511821382339, 0.012474783432306355, -0.01952327898718104], [-0.07598535674174661, -0.0, 0.0, -0.0], [0.17227890532278398, -0.0, 0.0, -0.0], [0.04814677429051867, -0.0, 0.0, -0.0], [0.0, 0.15695680524645847, 0.0, -0.0], [0.0, -0.08077576665905438, 0.0, -0.0], [0.0, 0.03809051929370203, 0.0, -0.0], [0.0, 0.0, -0.049517200018337475, 0.0], [0.0, -0.0, -0.04998638510222169, 0.0], [0.0, 0.0, -0.0497517925602796, 0.0], [-0.0, -0.0, 0.0, -0.07706820167421372], [-0.0, 0.0, 0.0, -0.09346895962372273], [-0.0, 0.0, 0.0, -0.08526858064896825], [0.046892364117130075, -5.393298302559541e-05, 0.06309520895098025, -0.09726350234072308], [0.004268497643369205, -0.027255614888658852, 0.10685021623650875, 0.09627472995081421], [-0.06274424281506634, 0.04906684091267763, -0.14231792826729353, 0.2124240665870351], [-0.04070569660679271, -0.11579104482091146, -0.014371045508180972, 0.2134713903143579], [-0.01065479530981447, 0.004212863334213925, 0.0007137447344568826, -0.0015943608866439746], [0.001829457743923176, 0.0022439310916146983, 0.0003455212685149066, 0.0042995670928666265], [-0.00014350411170401282, 0.00011108420211753875, 0.0012750500570991908, 0.003728852190276361], [-0.0004264438138611021, -0.01897076615565782, -0.004414438765750735, -0.0018197491322892055]], 'errors': {'features': 6.302339632288665e-16, 'ratios': 0.0, 'jacobian_five_point': 4.499455748617639e-12, 'optimized_jacobian_autograd': 1.3312082069802651e-16, 'optimized_target_value': 4.221032327516595e-17, 'optimized_target_gradient': 1.1214276511855276e-17, 'analytic_response': 7.424056130804059e-17}, 'target_value': -0.3151084124465187, 'target_gradient': [-0.0053485340539617746, -0.005477489513392514, -0.0004690641769977582, 0.010984313172502029], 'nonclosed_control_theta': [0.0, 0.7853981633974483], 'identity_control_gap': [0.0, 0.0], 'solver_control_seconds': 0.3861739999999969}]}


def fixture_family(planes=4):
    rng = np.random.default_rng(929173)
    n,m = 8,17*8
    rates = np.array([.04,.11,.23,.41,.67])
    components = rng.normal(size=(5,2,m))+1j*rng.normal(size=(5,2,m))
    times = np.arange(4)
    base = np.einsum('tk,ksm->tsm',np.exp(-times[:,None]*rates),components)
    derivative = np.einsum('tk,ksm->tsm',-rates*np.exp(-times[:,None]*rates),components)
    bands = np.stack([np.einsum('tk,ksn->tsn',np.exp(-times[:,None]*rates[np.arange(5)%4==band]),
        components[np.arange(5)%4==band,:,:n]) for band in range(4)],axis=2)
    p0 = np.exp(-2j*rng.uniform(-np.pi,np.pi,size=(4,4,n)))
    p1 = np.exp(-2j*rng.uniform(-np.pi,np.pi,size=(4,4,n)))
    valid = rng.uniform(size=(4,4,n))>.2
    return dict(base=base,base_derivative=derivative,base_band=bands,
        cos_basis=.1*(rng.normal(size=(planes,m))+1j*rng.normal(size=(planes,m))),
        sin_basis=.1*(rng.normal(size=(planes,m))+1j*rng.normal(size=(planes,m))),
        rates=np.linspace(.05,.5,planes),band_id=np.arange(planes)%4,
        phase0=p0,phase1=p1,valid=valid,
        reference_local_rms=float(np.sqrt(np.mean(np.abs(base[0].mean(axis=0)[:n])**2))),
        reference_global_rms=2.,global_weights=np.linspace(.01,.04,planes))


def covariance_oracle(a,b):
    x=np.column_stack([a.real,a.imag,b.real,b.imag])
    x-=x.mean(axis=0)
    c=x.T@x/len(x)
    va,vb=np.trace(c[:2,:2]),np.trace(c[2:,2:])
    re,im=c[0,2]+c[1,3],c[1,2]-c[0,3]
    fourth=np.mean(np.sum(x[:,:2]**2,axis=1)*np.sum(x[:,2:]**2,axis=1))
    return np.array([va,vb,re,(va+vb+2*re)/4,(va+vb-2*re)/4,re/np.sqrt(va*vb),
        (fourth-va*vb-2*np.sum(c[:2,2:]**2))/(va*vb),im/np.sqrt(va*vb)])


def test_analytic_moments_against_real_covariance_and_finite_difference():
    rng=np.random.default_rng(71844)
    a,b,da,db=rng.normal(size=(4,128))+1j*rng.normal(size=(4,128))
    value,derivative=moments_with_derivatives(a,b,da,db)
    assert np.max(np.abs(value-covariance_oracle(a,b)))<1e-12
    h=1e-3
    finite=(-covariance_oracle(a+2*h*da,b+2*h*db)+8*covariance_oracle(a+h*da,b+h*db)
            -8*covariance_oracle(a-h*da,b-h*db)+covariance_oracle(a-2*h*da,b-2*h*db))/(12*h)
    assert np.max(np.abs(finite-derivative)/(1+np.abs(derivative)))<1e-10


def test_time_derivatives_include_mean_centering():
    rng=np.random.default_rng(10483)
    a,b,da,db=rng.normal(size=(4,64))+1j*rng.normal(size=(4,64))
    stats,derivative=moments_with_derivatives(a,b,da,db)
    shifted,rate=moments_with_derivatives(a+13+2j,b-8j,da+7j,db-12)
    assert np.allclose(stats,shifted,atol=1e-12)
    assert np.allclose(derivative,rate,atol=1e-12)


def test_feature_layout_and_zero_plane_references():
    family=fixture_family(0)
    feature=features_numpy(family,np.zeros(0))
    assert feature.shape==(52,) and np.isfinite(feature).all()
    assert np.array_equal(ratios_numpy(family,np.zeros(0)),[0.,0.])
    n=family['valid'].shape[-1]
    values=family['base'][0]
    inc=values[:,n:].reshape(2,4,4,n)*family['phase1']-values[:,:n,None,None].transpose(0,2,3,1)*family['phase0']
    for lag in range(4):
        x=inc[:,lag][:,family['valid'][lag]]
        assert np.allclose(feature[lag*8:(lag+1)*8],covariance_oracle(x[0],x[1]),atol=1e-12)
    for band in range(4):
        a,b=family['base_band'][0,:,band]
        assert np.allclose(feature[32+band*3:35+band*3],[np.mean(np.abs(a)**2),np.mean(np.abs(b)**2),np.mean((a*b.conj()).real)],atol=1e-12)


def test_projection_quantization_and_scale_floors():
    assert np.array_equal(projection_indices('baseline'),np.arange(44))
    assert np.array_equal(projection_indices('capacity'),np.arange(48))
    assert np.array_equal(projection_indices('repair'),np.r_[np.arange(44),np.arange(48,52)])
    ref=np.arange(52,dtype=float)/10
    scale=robust_scales(np.stack([ref,ref,ref]))
    assert np.array_equal(scale,.001*np.maximum(1,np.abs(ref)))
    assert np.array_equal(quantize(ref,ref,scale),np.zeros(44,dtype=np.int64))
    assert np.array_equal(quantize(ref+.00049*scale,ref,scale),np.zeros(44,dtype=np.int64))
    assert np.array_equal(quantize(ref+.00051*scale,ref,scale),np.ones(44,dtype=np.int64))


def test_batched_numpy_and_ratios():
    family=fixture_family()
    theta=np.array([[.02,-.04,.01,.07],[-.03,.05,.06,-.02]])
    actual=features_numpy(family,theta,2)
    assert actual.shape==(2,52)
    assert np.allclose(actual,np.stack([features_numpy(family,row,2) for row in theta]),atol=1e-12)
    expected=np.sqrt(np.sum(2*(1-np.cos(theta))*family['global_weights'],axis=1))/family['reference_global_rms']
    assert np.allclose(ratios_numpy(family,theta)[:,1],expected,atol=1e-14)


def test_invalid_and_degenerate_inputs_rejected():
    family=fixture_family()
    for changed in ({**family,'rates':np.full(4,np.nan)}, {**family,'global_weights':-np.ones(4)},
                    {**family,'valid':np.zeros_like(family['valid'])}):
        try:
            features_numpy(changed,np.zeros(4))
        except ValueError:
            pass
        else:
            raise AssertionError('invalid family accepted')
    try:
        moments_with_derivatives(np.ones(8),np.ones(8),np.zeros(8),np.zeros(8))
    except ValueError:
        pass
    else:
        raise AssertionError('degenerate variance accepted')


def test_recorded_actual_gpu_features_and_jacobian():
    assert GPU_ATTESTATION is not None
    family=fixture_family()
    theta=np.array([.02,-.04,.01,.07])
    expected=features_numpy(family,theta)
    h=1e-3
    jac=np.column_stack([(-features_numpy(family,theta+2*np.eye(4)[j]*h)+8*features_numpy(family,theta+np.eye(4)[j]*h)
        -8*features_numpy(family,theta-np.eye(4)[j]*h)+features_numpy(family,theta-2*np.eye(4)[j]*h))/(12*h) for j in range(4)])
    for record in GPU_ATTESTATION['devices']:
        assert np.max(np.abs(np.asarray(record['features'])-expected)/(1+np.abs(expected)))<1e-10
        assert np.max(np.abs(np.asarray(record['ratios'])-ratios_numpy(family,theta)))<1e-10
        assert np.max(np.abs(np.asarray(record['jacobian'])-jac)/(1+np.abs(jac)))<1e-10
        assert abs(record['target_value']-features_numpy(family,theta,1)[22])<1e-10
        target_grad=np.array([(-features_numpy(family,theta+2*np.eye(4)[j]*h,1)[22]+8*features_numpy(family,theta+np.eye(4)[j]*h,1)[22]
            -8*features_numpy(family,theta-np.eye(4)[j]*h,1)[22]+features_numpy(family,theta-2*np.eye(4)[j]*h,1)[22])/(12*h) for j in range(4)])
        assert np.max(np.abs(np.asarray(record['target_gradient'])-target_grad)/(1+np.abs(target_grad)))<1e-10


def test_spectral_family_and_fixed_difference_with_hostile_controls():
    ell,m=harmonic_layout(48)
    rng=np.random.default_rng(8217)
    x=rng.normal(size=(2,2,len(ell)))+1j*rng.normal(size=(2,2,len(ell)))*(m>0)
    x[...,ell<30]=0
    planes=build_planes(x,24,shells=[30,35,40],planes_per_shell=2)
    value=apply_angles(x,planes,np.linspace(-.7,.7,6))
    assert invariants(x,value)['passed']
    planes['u'][0]*=1.01;planes['coadd_u'][0]*=1.01
    assert not invariants(x,apply_angles(x,planes,np.full(6,.5)))['passed']


def test_exact_nonclosure_and_redundant_constraint_projection():
    a,b=np.array([1,1,-1,-1]),np.array([1,-1,1,-1])
    assert np.mean(a*a)==np.mean(b*b) and np.mean(a**4)==np.mean(b**4)
    smooth=lambda x:(x+np.roll(x,-1))/2
    assert np.mean(smooth(a)**4)!=np.mean(smooth(b)**4)
    projected,inverse,rank=tangent_reference([[1.,0.],[2.,0.]],np.array([1.,2.]))
    assert rank==1 and np.allclose(projected,[0,2])
    assert matching_bins(np.array([2.]),np.array([2.]),np.ones(1),[0])['matched']
    assert np.array_equal(initial_kicks(24,'fresh')[0],initial_kicks(24,'fresh')[0])


def test_recorded_actual_projected_search_control():
    assert GPU_ATTESTATION is not None
    for device in GPU_ATTESTATION['devices']:
        point=np.asarray(device['nonclosed_control_theta'])
        assert abs(point[0])<.0005 and abs(point[1])>=.05
        assert np.max(np.abs(device['identity_control_gap']))<.001


def test_boundary_failure_is_not_a_representation_separation():
    ref=np.zeros(52);scales=np.ones(52);value=ref.copy()
    value[48]=.00049995
    assert not bin_separation(value,ref,scales,'repair')['robust_changed_coordinate']
    value[48]=.00050005
    assert not bin_separation(value,ref,scales,'repair')['robust_changed_coordinate']
    value[48]=.001
    assert bin_separation(value,ref,scales,'repair')['robust_changed_coordinate']
