_wins], axis=0)
    train_target = np.concatenate([[1]*len(pos_wins), [0]*len(neg_wins)], axis=0)#[..., None]
    train_target = np.stack([train_target, -(t