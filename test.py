import sys,argparse

# 从控制台接收参数
parser = argparse.ArgumentParser(description="Model Param Set")
parser.add_argument('--train',default=False,type=bool,help='to tain the Model ')
parser.add_argument('--drop_rate',default='0.3',type=float,help='to set drop out rate')
args = parser.parse_args()

if __name__ == '__main__':
    print(args)