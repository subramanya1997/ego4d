# def get_train_loader(dataset, video_features, configs):
#     train_set = Dataset(dataset=dataset, video_features=video_features)
#     train_loader = torch.utils.data.DataLoader(
#         dataset=train_set,
#         batch_size=configs.batch_size,
#         shuffle=True,
#         collate_fn=train_collate_fn,
#     )
#     return train_loader


# def get_test_loader(dataset, video_features, configs):
#     test_set = Dataset(dataset=dataset, video_features=video_features)
#     test_loader = torch.utils.data.DataLoader(
#         dataset=test_set,
#         batch_size=configs.batch_size,
#         shuffle=False,
#         collate_fn=test_collate_fn,
#     )
#     return test_loader