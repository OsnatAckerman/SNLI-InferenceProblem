STUDENT = {'name': "Osnat Ackerman_Shira Yair",
    'ID': '315747204_315389759'}
import torch
from torch import nn
import sys
from Model_SNLI import Input_representation
import load_data
import time
LR = 0.05
EPOCHS = 170
BATCH_SIZE = 32


def calc_accuracy(prediction, labels):
    good = bed = 0
    for p, l in zip(prediction, labels):
        if int(torch.argmax(p)) == int(l):
            good += 1
        else:
            bed += 1
    return good / (good + bed)


def validation_check(i, model, valid_loader, loss_func, device, cud):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    print(f'Epoch: {i + 1:02} | Starting Evaluation...')
    for batch in valid_loader:
        prediction, loss = apply(model, loss_func, batch, device, cud)
        epoch_acc += calc_accuracy(prediction, batch.label)
        epoch_loss += loss.item()
    print(f'Epoch: {i + 1:02} | Finished Evaluation')
    return float(epoch_loss) / len(valid_loader), float(epoch_acc) / len(valid_loader)


def apply(model, criterion, batch, device, cud):
    sentence_a, length_a = batch.premise
    sentence_b, length_b = batch.hypothesis
    label = batch.label
    sentence_a, length_a, sentence_b, length_b, label = sentence_a.to(device), length_a.to(device), \
                                                        sentence_b.to(device), length_b.to(device), label.to(device)
    if cud:
        sentence_a, length_a, sentence_b, length_b, label = sentence_a.cuda(), length_a.cuda(), sentence_b.cuda(), \
                                                            length_b.cuda(), label.cuda()
    pred = model(sentence_a, sentence_b, length_a, length_b)
    loss_ = criterion(pred, label)
    return pred, loss_


def train_and_eval_model(model, text_field, train_loader, valid_loader, loss_func, device, cud):
    # with open('./train_accuracy.json', 'w') as train_writer:
    #     with open('./dev_accuracy.json', 'w') as dev_writer:
    torch.manual_seed(3)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LR, initial_accumulator_value=0.1, weight_decay=1e-05)
    for i in range(EPOCHS):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        print(f'Epoch: {i + 1:02} | Starting Training...')
        for batch in train_loader:
            optimizer.zero_grad()
            model.zero_grad()
            prediction, loss = apply(model, loss_func, batch, device, cud)
            epoch_acc += calc_accuracy(prediction, batch.label.to(device))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {i + 1:02} | Finished Training')
        avg_epoch_loss, avg_epoch_acc = float(epoch_loss) / len(train_loader), float(epoch_acc) / len(train_loader)
        avg_epoch_loss_val, avg_epoch_acc_val = validation_check(i, model, valid_loader, loss_func, device,
                                                                 cud)
        # train_writer.write(f'{avg_epoch_acc * 100:.2f}\n')
        # dev_writer.write(f'{avg_epoch_acc_val * 100:.2f}\n')
        print(f'\tTrain Loss: {avg_epoch_loss:.3f} | Train Acc: {avg_epoch_acc * 100:.2f}%')
        print(f'\t Val. Loss: {avg_epoch_loss_val:.3f} |  Val. Acc: {avg_epoch_acc_val * 100:.2f}%')
    return model


def test_accuracy(model, test_loader, loss_func, device, cud):
    _loss = 0
    acc = 0
    model.eval()
    print(f'Starting test Evaluation...')
    for batch in test_loader:
        prediction, loss = apply(model, loss_func, batch, device, cud)
        acc += calc_accuracy(prediction, batch.label)
        _loss += loss.item()
    # print(f'accuracy on dev: {i + 1:02} | Finished Evaluation')
    test_loss,test_acc =  float(_loss) / len(test_loader), float(acc) / len(test_loader)
    print(f'\t Test. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')



if __name__ == "__main__":
    cud = torch.cuda.is_available()
    device = torch.device("cuda" if cud else "cpu")
    (snli_train_iter, snli_val_iter, snli_test_iter), TEXT_FIELD, LABEL_FIELD = load_data.read_data(BATCH_SIZE)
    model = Input_representation(3, TEXT_FIELD, device, cud).to(device)
    if cud:
        model = model.cuda()
    model = train_and_eval_model(model, TEXT_FIELD, snli_train_iter, snli_val_iter, nn.CrossEntropyLoss(),
                                 device, cud)
    test_accuracy(model, snli_test_iter,nn.CrossEntropyLoss(), device, cud)


