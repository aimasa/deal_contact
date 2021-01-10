=====================================
关系提取部分
=====================================
使用的关系抽取模型及相关论文信息
@article{wang2019extracting,
  title={Extracting Multiple-Relations in One-Pass with Pre-Trained Transformers},
  author={Wang, Haoyu and Tan, Ming and Yu, Mo and Chang, Shiyu and Wang, Dakuo and Xu, Kun and Guo, Xiaoxiao and Potdar, Saloni},
  journal={arXiv preprint arXiv:1902.01030},
  year={2019}
}
代码运行文件：run_classifier.py
data_dir =>包括train、test、dev数据集的文件夹（还有需要预测的label的txt，我这边会给一个文件夹，解压后，文件夹路径填这就行
ps：do_eval这个功能没有用处，代码是逻辑错误的
do_predict这个功能我添加了对准确度的输出
vocab_file是bert预训练模型的总路径
max_seq_length 是最大句子长度
train_batch_size 是训练时，一次性输入的句子的个数大小
learning_rate 学习速率，控制反向传播的梯度下降的快慢
max_distance 我也不知道是什么参数，所以我没碰过
num_train_epochs 训练的迭代次数
max_num_relations 一个句子中最多有存在多少关系
output_dir 结果输出路径
init_checkpoint 初始化模型位置
python run_classifier.py \
        --task_name=semeval \
        --do_train=true \
        --do_eval=false \
        --do_predict=false \
        --data_dir=$DATA_DIR/semeval2018/multi \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=4 \
        --learning_rate=2e-5 \
        --num_train_epochs=30 \
        --max_distance=2 \
        --max_num_relations=12 \
        --output_dir=<path to store the checkpoint>