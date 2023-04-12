from CausalTAD import Trainer as CausalTAD

# training
causalTAD = CausalTAD(save_model="test", load_model=None, city="chengdu")
causalTAD.train()


# inference
causalTAD = CausalTAD(save_model=None, load_model="test_10", city="chengdu")
causalTAD.test()