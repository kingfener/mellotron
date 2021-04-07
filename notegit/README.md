# note
笔记


## 说明
  本Git用来记录笔记

# cmd
  sudo git remote remove notegit 




# git ref
  1, https://www.jianshu.com/p/d42d330bfead
    简化git subtree命令

    我们已经知道了git subtree 的命令的基本用法，但是上述几个命令还是显得有点复杂，特别是子仓库的源仓库地址，特别不方便记忆。
    这里我们把子仓库的地址作为一个remote，方便记忆：
    ref: https://www.jianshu.com/p/d42d330bfead
    git remote add -f taco_sma git@github.com:kingfener/Attentions-in-Tacotron.git
    git subtree add --prefix=taco_sma taco_sma master --squash
    git subtree pull --prefix=taco_sma taco_sma master --squash
    git subtree push --prefix=taco_sma taco_sma master

    作者：好好编程
    链接：https://www.jianshu.com/p/d42d330bfead
    来源：简书
    著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
    
  # 
    git remote add -f notegit git@github.com:kingfener/note.git
    然后可以这样来使用git subtree命令：
    git subtree add --prefix=notegit notegit main --squash
    git subtree pull --prefix=notegit notegit main --squash
    git subtree push --prefix=notegit notegit main


# git config
  git config --global user.name king
  git config --global user.email kingfener@gmail.com

#

# pip install 
  pip --default-timeout=100 install --upgrade -i https://mirrors.aliyun.com/pypi/simple tensorflow==1.15.0
