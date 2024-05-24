
import argparse
from src.run import train,test,infer,generate
from src.utils import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=['train', 'test', 'infer','generate'])
parser.add_argument('--data_input_path',  type=str,default = 'data')
parser.add_argument('--model_input_path',  type=str, default = "microsoft/biogpt")    
parser.add_argument('--output_path',  type=str, default = 'outputs/critical')  
parser.add_argument('--outcome',  type=str, default = "outcome_critical",choices=['outcome_ed_los','outcome_critical'])
parser.add_argument('--batch_size', type=int, default=128)  
parser.add_argument('--buffer_steps', type=int, default=2048)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--ppo_epochs', type=int, default=10)  
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--max_grad_norm', type=float, default=1.0)  

parser.add_argument('--total_timesteps', type=float, default=20000)  
parser.add_argument('--penalty_ratio', type=int, default=15)
parser.add_argument('--wrong_prediction_penalty', type=int, default=99)
parser.add_argument("--random_seed", type=int, default=42)
args = parser.parse_args()
set_random_seed(args.random_seed)

if __name__ == "__main__":
    if args.mode == 'train':
        print('-----------train------------')
        train(args)

    if args.mode == 'test':
        print('-------------test--------------')
        test(args)

    if args.mode == 'infer':
        print('-------------infer--------------')
        infer(args)
        
    if args.mode == 'generate':
        print('-------------generate--------------')
        generate(args)