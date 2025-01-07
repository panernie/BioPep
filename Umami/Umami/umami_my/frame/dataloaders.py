import torch.utils.data as data

from Umami.classification.umami_my.frame.att_data import AttDataset, BertDataset


def att_data_loader(u_data=None, label=None, batch_size=None, shuffle=None,
                    num_workers=0, is_bert=False, is_fixed=True, collate_fn=BertDataset.collate_fn,
                    collate_fn2=BertDataset.collate_fn2):
    if not is_bert:
        dataset = AttDataset(u_data, label)
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers
                                      )
    else:
        dataset = BertDataset(u_data, label)
        if is_fixed:
            data_loader = data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          collate_fn=collate_fn2)
        else:
            data_loader = data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          collate_fn=collate_fn)
    return data_loader
