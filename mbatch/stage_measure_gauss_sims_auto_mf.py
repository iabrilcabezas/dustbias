

# load data sim

mf_data = autils.compute_meanfield(est1, mask, args)

        xy_g0 = xy_g - mf_data['mf_grad_set0']
        xy_g1 = xy_g - mf_data['mf_grad_set1']