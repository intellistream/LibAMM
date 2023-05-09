import torch
def main():
    #load the library, assume it is located together with this file
    torch.ops.load_library("../libIntelliStream.so")
    #gen the input tensor
    a=torch.randn(2,10)
    b=torch.randn(2,10)
    #The pytorch +
    print('/****test add****/')
    print('pytorch+:',a+b)
    #our c++ extension of +
    print('myLib+:',torch.ops.myLib.myVecAdd(a,b))
     #The pytorch -
    print('/****test sub****/')
    print('pytorch-:',a-b)
    #our c++ extension of -
    print('myLib-:',torch.ops.myLib.myVecSub(a,b))
 
  

if __name__ == "__main__":
    main()