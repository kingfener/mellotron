



# git subtree
    ref: https://www.jianshu.com/p/d42d330bfead
    git remote add -f taco_sma git@github.com:kingfener/Attentions-in-Tacotron.git
    git subtree add --prefix=taco_sma taco_sma master --squash
    git subtree pull --prefix=taco_sma taco_sma master --squash
    git subtree push --prefix=taco_sma taco_sma master



# snap install 

    sudo snap install picgo_2.2.2_amd64.snap --dangerous


# 
pip --default-timeout=100 install --upgrade -i https://mirrors.aliyun.com/pypi/simple tensorflow==1.15.0

git remote add -f notegit git@github.com:kingfener/note.git

然后可以这样来使用git subtree命令：
git subtree add --prefix=notegit notegit master --squash
git subtree pull --prefix=notegit notegit master --squash
git subtree push --prefix=notegit notegit master
