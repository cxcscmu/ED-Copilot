
import argparse
import os
from src.utils import set_random_seed
from src.run_sft import train,test


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=['train', 'test'])
parser.add_argument('--data_input_path',  type=str,default = 'data')
parser.add_argument('--model_input_path',  type=str, default = "microsoft/biogpt")    
parser.add_argument('--output_path',  type=str, default = 'outputs/critical')  
parser.add_argument('--outcome', type=str, default = "outcome_critical",choices=['outcome_ed_los','outcome_critical']) 
parser.add_argument('--baseline', type=bool, default=False)

parser.add_argument('--train_batch_size', type=int, default=8)  
parser.add_argument('--valid_batch_size', type=int, default=128)  
parser.add_argument('--test_batch_size', type=int, default=128) 

parser.add_argument('--epochs', type=int, default=15)  
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--warmup_percent', type=float, default=0.1)   
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--class_weight', type=int, default=10)
parser.add_argument("--random_seed", type=int, default=42)

args = parser.parse_args()
args.local_rank = int(os.environ.get("LOCAL_RANK",-1))

if __name__ == "__main__":
    set_random_seed(args.random_seed)

    if args.mode == 'train':
        if args.local_rank in [-1,0]:
            print('-----------train------------')
        train(args)

    if args.mode == 'test':
        print('-------------test--------------')
        ################## You should use single GPU for testing. ####################
        assert args.local_rank == -1
        test(args)
