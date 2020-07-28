import entity_contract.normal_param as normal_param
import entity_contract.NER_pre_data as predata
from model.LSTM_CRF import BiLSTM_CRF
import torch.optim as optim

def train(head_path):
    # word_to_ix = creat_vocab(head_path)
    tag_to_ix = predata.build_label(normal_param.labels)
    model : BiLSTM_CRF = BiLSTM_CRF(len(word_to_ix), tag_to_ix, normal_param.EMBEDDING_DIM, normal_param.HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    print()
    # for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    #     for sentence, tags in training_data:
    #         # Step 1. Remember that Pytorch accumulates gradients.
    #         # We need to clear them out before each instance
    #         model.zero_grad()
    #
    #         # Step 2. Get our inputs ready for the network, that is,
    #         # turn them into Tensors of word indices.
    #         sentence_in = LSTM_CRF.prepare_sequence(sentence, word_to_ix)
    #     #         targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
    #
    #         # Step 3. Run our forward pass.
    #         loss = model.neg_log_likelihood(sentence_in, targets)
    #
    #         # Step 4. Compute the loss, gradients, and update the parameters by
    #         # calling optimizer.step()
    #         loss.backward()
    #         optimizer.step()