import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable
from fabulous.color import fg256


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight)
        m.bias.data.fill_(0.01)


def vector_difference(x1, x2):
    x1_n = x1 / (torch.norm(x1, p=2, dim=1, keepdim=True)+1e-6)
    x2_n = x2 / (torch.norm(x2, p=2, dim=1, keepdim=True)+1e-6)
    return x1_n - x2_n


def pcc_ccc_loss(labels_th, scores_th):
    std_l_v = torch.std(labels_th[:,0]); std_p_v = torch.std(scores_th[:,0])
    std_l_a = torch.std(labels_th[:,1]); std_p_a = torch.std(scores_th[:,1])
    mean_l_v = torch.mean(labels_th[:,0]); mean_p_v = torch.mean(scores_th[:,0])
    mean_l_a = torch.mean(labels_th[:,1]); mean_p_a = torch.mean(scores_th[:,1])
    
    PCC_v = torch.mean( (labels_th[:,0] - mean_l_v) * (scores_th[:,0] - mean_p_v) ) / (std_l_v * std_p_v)
    PCC_a = torch.mean( (labels_th[:,1] - mean_l_a) * (scores_th[:,1] - mean_p_a) ) / (std_l_a * std_p_a)
    CCC_v = (2.0 * std_l_v * std_p_v * PCC_v) / ( std_l_v.pow(2) + std_p_v.pow(2) + (mean_l_v-mean_p_v).pow(2) )
    CCC_a = (2.0 * std_l_a * std_p_a * PCC_a) / ( std_l_a.pow(2) + std_p_a.pow(2) + (mean_l_a-mean_p_a).pow(2) )
    
    PCC_loss = 1.0 - (PCC_v + PCC_a)/2
    CCC_loss = 1.0 - (CCC_v + CCC_a)/2
    return PCC_loss, CCC_loss, CCC_v, CCC_a


def pair_mining(inp1, inp2, fixed_sample, is_positive):
    sim_matrix = inp1 @ inp2.t()

    if fixed_sample:  # Ranking-based mining
        batch_size = inp2.size(0)
        sort_size = int(0.8 * batch_size)

        if is_positive:
            _, ind = torch.topk(sim_matrix, k=sort_size, dim=0, largest=True)
        else:
            _, ind = torch.topk(sim_matrix, k=sort_size, dim=0, largest=False)
    
        sort_ind = ind[:,0]
        inp1_sort = inp1[sort_ind]
        inp2_sort = inp2[sort_ind]
        return inp1_sort, inp2_sort
    else:  
        # Pair mining from metric learning.
        # Because of absense of categorical label, we slightly tuned original pair mining.
        epsilon = 0.0005
        if is_positive:
            value, _ = torch.max(sim_matrix, dim=0)
            ind = sim_matrix < (value-epsilon)
        else:
            value, _ = torch.min(sim_matrix, dim=0)
            ind = sim_matrix > (value+epsilon)

        sort_ind = []
        for i in range(len(ind[0])):
            prob = sum(ind[i]) / len(ind[i])
            # Select informative pairs for constructing positive or negative pairs as probability values.
            if prob >= 0.7:
                sort_ind.append(i)

        inp1_sort = inp1[sort_ind]
        inp2_sort = inp2[sort_ind]
        return inp1_sort, inp2_sort


def penalty_function(inp1, inp2):
    return 0.5 * torch.ones_like(inp1[:,0]) * (torch.sign(inp1[:,0]) != torch.sign(inp2[:,0]))


def encoder_fun(models, feats):
    fc_mu, fc_logvar = models[0], models[1]
    mu = fc_mu(feats)
    logvar = fc_logvar(feats)
    return mu, logvar
   
 
def reparameterize(mu, logvar, training):
    if training:
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return torch.add(torch.mul(std, eps), mu)
    else:
        return mu


def CLIP_Encoding(inputs, all_z, clip_enc, idn_list):

    import clip
    gender_list     = ["women", "men"]  # ["female", "male"]
    skin_color_list = ["white", "yellow", "black"]    # skin
    age_group_list  = ["young", "middle-age", "old"]
    view_list       = ["forward", "side"]

    # Appearance encoding
    z_app = clip_enc.encode_image(inputs).float()
    z_app /= z_app.norm(dim=-1, keepdim=True)  # [BS, 512]

#    # Individual encoding
#    idn_texts = clip.tokenize(idn_list).cuda()
#    z_idn = clip_enc.encode_text(idn_texts).float()
#    z_idn /= z_idn.norm(dim=-1, keepdim=True)  # [BS, 512]

    # Gender encoding
    gender_txt = torch.cat([clip.tokenize(f"a photo of a {c}") for c in gender_list]).cuda()
    gender_enc = clip_enc.encode_text(gender_txt).float()  # [2, 512]
    gender_enc /= gender_enc.norm(dim=-1, keepdim=True)
    similarity = (100. * z_app @ gender_enc.T).softmax(dim=-1)  # [BS, 2]
    _, indexes_gnd = similarity.topk(1)
    z_gnd = gender_enc[indexes_gnd.squeeze()]
    z_gnd /= z_gnd.norm(dim=-1, keepdim=True)  # [BS, 512]

    # Skin color encoding
    skin_txt = torch.cat([clip.tokenize(f"a photo with {c} skin color") for c in skin_color_list]).cuda()
    skin_enc = clip_enc.encode_text(skin_txt).float()
    skin_enc /= skin_enc.norm(dim=-1, keepdim=True)
    similarity = (100. * z_app @ skin_enc.T).softmax(dim=-1)
    _, indexes_skn = similarity.topk(1)
    z_skn = skin_enc[indexes_skn.squeeze()]

    # Age encoding
    age_txt = torch.cat([clip.tokenize(f"a facial photo of in one's {c} group") for c in age_group_list]).cuda()
    age_enc = clip_enc.encode_text(age_txt).float()
    age_enc /= age_enc.norm(dim=-1, keepdim=True)
    similarity = (100. * z_app @ age_enc.T).softmax(dim=-1)
    _, indexes_age = similarity.topk(1)
    z_age = age_enc[indexes_age.squeeze()]

    # Face view encoding
    viw_txt = torch.cat([clip.tokenize(f"a facial photo facing {c}") for c in view_list]).cuda()
    viw_enc = clip_enc.encode_text(viw_txt).float()
    viw_enc /= viw_enc.norm(dim=-1, keepdim=True)
    similarity = (100. * z_app @ viw_enc.T).softmax(dim=-1)
    _, indexes_viw = similarity.topk(1)
    z_viw = viw_enc[indexes_viw.squeeze()]

#    # -> Answer
#    gender_txt = torch.cat([clip.tokenize(f"a photo of a {gender_list[indexes[i]]}") for i in range(z_app.size(0))]).cuda()
#    z_gnd = clip_enc.encode_text(gender_txt).cuda()

    return z_app, z_gnd, z_skn, z_age, z_viw


def penalty_function(inp1, inp2):
    return 0.5 * torch.ones_like(inp1[:,0]) * (torch.sign(inp1[:,0]) != torch.sign(inp2[:,0]))


def LENGTH_CHECK(vec):
    if len(vec) == 1:
        return torch.cat([vec, vec], dim=0)
    else:
        return vec


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def Cross_Attention(opt, model, vectors):

    CHA = model
    shift_list = []
    no_domain = len(vectors)
    anchor_vec = vectors[0]
    
    for user in range(no_domain-1):  # one-to-N pair-based cross attention
        weights = CHA(anchor_vec.unsqueeze(0), vectors[user+1].unsqueeze(0))
        shift_list.append(weights)
        
    return shift_list


def Sinkhorn_Knopp(opt, vectors, metric, sinkhorn=True):

    cos = metric
    no_domain = len(vectors)
    mean = [torch.mean(vectors[i], dim=0, keepdim=True) for i in range(no_domain)]
    
    # weights of OT
    sim_user_anchor = torch.zeros(size=(no_domain-1, vectors[0].size(0),1))
    sim_user_list = []
    for user in range(no_domain-1):
        sim_user_list.append( torch.zeros(size=(1,vectors[user+1].size(0),1)) )
    anchor_vec = vectors[0]
    anchor_mean = mean[0]
    
    if opt.relevance_weighting == 0:
        for user in range(no_domain-1):
            gen = (vector.norm(p=2) for vector in anchor_vec)
            for idx, sim in enumerate(gen):
                sim_user_anchor[user,idx,0] = sim
            gen = (vector.norm(p=2) for vector in vectors[user+1])
            for idx, sim in enumerate(gen):
                sim_user_list[user][0,idx,0] = sim
    elif opt.relevance_weighting == 1:
        for user in range(no_domain-1):
            gen = (vector @ mean[user+1].t() for vector in anchor_vec)
            for idx, sim in enumerate(gen):
                sim_user_anchor[user,idx,0] = F.relu(sim)
            gen = (vector @ anchor_mean.t() for vector in vectors[user+1])
            for idx, sim in enumerate(gen):
                sim_user_list[user][0,idx,0] = F.relu(sim)

    sim_user_anchor = (sim_user_anchor.size(1)*sim_user_anchor) / (torch.sum(sim_user_anchor,1).unsqueeze(1)+opt.epsilon)
    for user in range(no_domain-1):
        sim_user_list[user] = (sim_user_list[user].size(1)*sim_user_list[user]) / (torch.sum(sim_user_list[user],1).unsqueeze(1))

    # cost of OT
    cos_mat_list = []
    for user in range(no_domain-1):
        cos_mat_list.append(torch.zeros(size=(1,vectors[0].size(0),vectors[user+1].size(0))))
    for user in range(no_domain-1):
        for left in range(cos_mat_list[user].size(1)):
            for right in range(cos_mat_list[user].size(2)):
                cos_mat_list[user][0,left,right] = 1. - cos(vectors[0][left].unsqueeze_(0), vectors[user+1][right].unsqueeze_(0))


    if sinkhorn:
        _lambda, _scale_factor, _no_iter = 5., 0.1, 5
        scale_list, shift_list = [], []
        for user in range(no_domain-1):  # repeat for each identity
            r = sim_user_anchor[user]
            c = sim_user_list[user].squeeze(0)
            M = cos_mat_list[user].squeeze(0)
            
            u = torch.ones(size=(r.size(0),1))/r.size(0)
            K = torch.exp(-_lambda*M)
            K_tilde = torch.diag(1/(r[:,0]+opt.epsilon)) @ K
            
            # update u,v
            for itrs in range(_no_iter):
                u_new = 1 / ( K_tilde @ (c/(K.t()@u+opt.epsilon))+opt.epsilon )
                u = u_new
            v = c / (K.t()@u+opt.epsilon)

            apprx_flow = torch.diag(u.squeeze(1)) @ K @ torch.diag(v.squeeze(1))
            MMM = apprx_flow * M
            mu_e = torch.sum(MMM, dim=0, keepdim=False).unsqueeze(1)
            shift_list.append(mu_e.detach().cuda())
        return shift_list
