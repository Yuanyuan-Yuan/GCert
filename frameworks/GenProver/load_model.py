############################################################
# This scripts shows how to load your customized model     #
# (see examples in `experiments/model.py`) such that it    #
# can fit into GenProver/ExactLine.                        #
############################################################

import genmodels

def convert_key(num_seq, state_dict):
    for key in list(state_dict.keys()):
        for i in range(num_seq):
            if 'seq_%d' % i in key:
                new_s = '%d.net' % i
                state_dict[key.replace('seq_%d' % i, new_s)] = state_dict.pop(key)
                break
    return state_dict

gen_state_dict = gen_ckpt['generator']
gen_num_seq = 4 # For `ConvGeneratorSeq32` in `experiments/model.py`
gen_state_dict = convert_key(gen_num_seq, gen_state_dict)


cls_state_dict = cls_ckpt['classifier']
cls_num_seq = 7 # For `F3` and `F4` in `experiments/model.py`
cls_state_dict = convert_key(cls_num_seq, cls_state_dict)

###########################################################
# Note that the initialization is different with Pytorch. #
###########################################################
generator = genmodels.ConvGenerator().infer([50]).to(h.device)
generator.eval()

classifier = genmodels.F3().infer([1, 32, 32]).to(h.device)
classifier.eval()

decoder.load_state_dict(gen_state_dict)
classifier.load_state_dict(cls_state_dict)