def train_val_loop():
    train_acc_history = []
    val_loss_history = []
    model.train()
    for epoch in tqdm(range(10)):
        train_loss = 0
        train_acc = 0
        for X_batch, y_batch in train_dataloader:
            y_batch_pred = model(X_batch)
            loss = loss_fn(y_batch_pred, y_batch)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc += torch.sum(y_batch ==
                                   torch.argmax(y_batch_pred, dim=-1))/BATCH_SIZE
            batch_cnt += 1
        train_acc /= batch_cnt
        train_acc_history += [train_acc]
        train_loss /= batch_cnt
        train_loss_history += [train_loss]

        model.eval()
        with torch.no_grad():
            val_acc = 0
            val_loss = 0
            val_batch_cnt = 0
            for X_batch, y_batch in val_dataloader:
                y_batch_pred = model(X_batch)
                loss = loss_fn(y_batch_pred, y_batch)
                val_loss += loss.item()
                val_acc += torch.sum(y_batch ==
                                     torch.argmax(y_batch_pred, dim=-1))/BATCH_SIZE
                val_batch_cnt += 1
            val_acc /= val_batch_cnt
            val_acc_history += [val_acc]
            val_loss /= val_batch_cnt
            val_loss_history += [val_loss]
            sim_test_words(test_words, word2idx, model)
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history