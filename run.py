from utils import *


from embedding import *
from forward import *
from my_attention import *
from encode import *
from decode import *            
from my_input import *

def make_model(source_vocab,target_vocab,N=6,d_model=512,d_ff=2048,head=8,dropout = 0.2):
    """
    This function constructs a Transformer model with specified parameters.
    
    Parameters:
        source_vocab (int): The size of the source vocabulary.
        target_vocab (int): The size of the target vocabulary.
        N (int): The number of layers in the encoder and decoder.
        d_model (int): The dimensionality of the model (embedding dimension).
        d_ff (int): The dimensionality of the feed-forward layers.
        head (int): The number of attention heads.
        dropout (float): The dropout rate for regularization.

    Returns:
        model: A complete Transformer model instance.
    """

    c = copy.deepcopy
    
    # Initialize multi-head attention, feed-forward layers, and positional encodings
    attn = MultiHeadedAttention(head,d_model)

    ff = PositionwiseFeedForward(d_model,d_ff,dropout)

    position = PositionalEncoding(d_model,dropout)
    
    # Create the encoder-decoder architecture using the initialized components
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
        Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout),N),
        nn.Sequential(Embeddings(d_model,source_vocab),c(position)),
        nn.Sequential(Embeddings(d_model,target_vocab),c(position)),
        Generator(d_model,target_vocab)
    )

    # Initialize the model parameters using Xavier uniform distribution
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def gpu_data_generator(source,target,batch_size):
    """
    Converts source and target data into tensors suitable for GPU processing.
    
    Parameters:
        source (list): List of source strings.
        target (list): List of target strings.
        batch_size (int): The size of each batch for processing.

    Returns:
        source_data (Tensor): Source data as a tensor.
        target_data (Tensor): Target data as a tensor.
        length (int): The length of the dataset.
    """
    if len(source) != len(target):
        return 

    length = len(source)
    if batch_size > 1:
        source += source[:batch_size]
        target += target[:batch_size]

    source = [[ord(char) for char in source_i] for source_i in source]
    target = [[ord(char)  for char in target_i ] for target_i in target ]
    source_data = torch.LongTensor(source)
    target_data = torch.LongTensor(target)
    
    if torch.cuda.is_available():
        source_data = source_data.to(device)
        target_data = target_data.to(device)

    return source_data,target_data,length
    
        

def data_generator(source_data,target_data,length,batch_size):
    """
    Generates batches of data for training.

    Parameters:
        source_data (Tensor): Source data tensor.
        target_data (Tensor): Target data tensor.
        length (int): The length of the dataset.
        batch_size (int): The batch size for training.

    Yields:
        Batch: A batch of source and target data.
    """
    for i in range(length):
        # src = Variable(source_data[i].unsqueeze(0),requires_grad=False)
        # trg = Variable(target_data[i].unsqueeze(0),requires_grad=False)
        src = Variable(source_data[i:i + batch_size],requires_grad=False)
        trg = Variable(target_data[i:i + batch_size],requires_grad=False)
        yield Batch(src,trg)
                
def run(model,loss,source,target,test_data,batch_size,epochs=5):
    """
    Runs the training loop and performs greedy decoding on the test data.
    
    Parameters:
        model: The Transformer model.
        loss: The loss function.
        source (list): List of source strings for training.
        target (list): List of target strings for training.
        test_data (str): The test input string for evaluation.
        batch_size (int): The batch size for training.
        epochs (int): Number of epochs for training.
    
    Returns:
        best_model_params: The best model parameters based on the lowest training loss.
    """
    i = 0

    best_loss = 1000

    source_data,target_data,length = gpu_data_generator(source,target,batch_size)
    for epoch in range(epochs):

        model.train()

        # Train the model for one epoch
        train_loss = run_epoch(data_generator(source_data,target_data,length,batch_size),model,loss)
        print(train_loss)


        if train_loss < best_loss:
            best_loss = train_loss
            print('Saving the best model parameters')

            torch.save(model,'test.pth')

        model.eval()

        run_epoch(data_generator(source_data,target_data,length,batch_size),model,loss)

        print(f"Epoch {i} finished with loss {train_loss}")
        i += 1

    model.eval()

    test_data = [ord(char)  for char in test_data ]
    source = Variable(torch.LongTensor(test_data).unsqueeze(0))

    source_mask = None

    max_len = source.size(-1)

    result = greedy_decode(model,source,source_mask,max_len,start_symbol = 101)
    

    source_list = [chr(ascii_item) for ascii_item in source[0]]
    target_list = [chr(ascii_item) for ascii_item in result[0]]
    best_model_params = torch.load('test.pth')
    print('input',source_list)
    print('output',target_list)
    return best_model_params

def eval(model,test_data):
    """
    Evaluates the model on the provided test data.
    
    Parameters:
        model: The trained model.
        test_data (str): The test string to evaluate.

    Returns:
        target_list (list): The output string predicted by the model.
    """
    if test_data in lost_res['source']:
        return eval_error(test_data)

    test_data = [ord(char)  for char in test_data]
    source = Variable(torch.LongTensor(test_data).unsqueeze(0))
    if torch.cuda.is_available():
        source = source.to(device)
    source_mask = None
    
    max_len = source.size(-1)
    result = greedy_decode(model,source,source_mask,max_len,start_symbol = 101)
    
    source_list = [chr(ascii_item) for ascii_item in source[0]]
    target_list = [chr(ascii_item) for ascii_item in result[0]]

    # print('input',source_list)
    # print('output',target_list)
    return target_list

def mapping(source,model):
    """
    Maps the source data to a target format using the model.
    
    Parameters:
        source (list): The input source data to be mapped.
        model: The trained model for performing the mapping.

    Returns:
        target (list): The mapped target data.
    """
    temp = False
    if source[1] == '-' and source[2] == '|':
        source = source[3:]
        source = ['s'] + source + [' '] * 2
        temp = True
    target = ['s'] + [' '] * (len(source) - 1)
    for i in range(1,len(source)):
        if source[i] == '+' or source[i] == '|' or source[i] == '-':
            target[i] = source[i]
        elif source[i] == ' ':
            continue
        else:
            target[i] = base_data[i - 1]
    mid_target = copy.copy(target)
    
    midput = eval(model,mid_target)
    midput[0] = 's'

    target = ['s'] + [' '] * (len(source) - 1)
    for i in range(1,len(midput)):
         if midput[i] == '+' or midput[i] == '|' or midput[i] == '-' or midput[i] == '0' or midput[i] == '*' or midput[i] == '1' or midput[i] == '?':
             target[i] = midput[i]
         elif midput[i] == ' ':
            continue
         else:
            target[i] = source[mid_target.index(midput[i])]
    if temp:
        target = ['s'] +['-','|'] + target[1:-2]
    return target

'''
input: 2 + 1 | 2 | 1 + 1 
output: 1 0 | 2 | 2 | 2
'''
def compute(source,model):
    """
    Computes the result by processing the source data through multiple steps using the model.
    
    Parameters:
        source (list): The input source data.
        model: The trained model used for computation.

    Returns:
        sum (list): The computed output.
    """
    mid_input1 = char_num_map(source)
    sum = []

    while True:
        if judge_end(mid_input1):
            break

        mid_input2 = eval(model,mid_input1)
        mid_input2[0] = 's'

        mid_input3 = eval(model,compare(mid_input1,mid_input2,source))
        mid_input4 = deal_base(mid_input3)
        sum =  mid_input4 + sum
        mid_input1 = copy.copy(mid_input2)

    sum += [' '] * (len(source) - len(sum))
    sum[0] = 's'
    sum = make_prograss(sum)

    return sum 

def align(source,model):
    mid_put0 = copy.copy(source)
    mid_put1 = []
    mid_put2 = []
    for i in range(0,12):
        mid_put1 = mapping(mid_put0,model)
        mid_put2 = compute(mid_put1,model)
        if mid_put2 == mid_put0:
            break
        mid_put0 = copy.copy(mid_put2)

    result = mid_put2



    # result = ['s', '1', '0', ' 0', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    res = ''

    # -| -3| -3| -3 
    result = result[1:]
    if (result[0] == '-' or result[0] == '0') and result[2] == '-':
        for i in range(0,len(result)):
            if result[i] == ' ':
                break
            if result[i] == '|' or result[i] == '-':
                continue
            else:
                res += result[i]
        if result[0] == '0':
            res = '-' + res
    # -3| -3| -3| -3 
    elif result[0] == '-' and result[3] == '-':
        res += '-'
        for i in range(0,len(result)):
            if result[i] == ' ':
                break
            if result[i] == '|' or result[i] == '-':
                continue
            else:
                res += result[i]
    # - |3| 3| 3 
    elif result[0] == '-' and result[3] == '|':
        res += '-'
        for i in range(0,len(result)):
            if result[i] == ' ':
                break
            if result[i] == '|' or result[i] == '-':
                continue
            else:
                res += result[i]
    else:
        for i in range(0,len(result)):
            if result[i] == ' ':
                break
            elif result[i] == '|':
                continue
            else:
                res += result[i]
# 3| 3 3 3 3 
    return res

