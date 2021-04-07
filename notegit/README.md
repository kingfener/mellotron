# note
笔记


## 说明
  本Git用来记录笔记




# git ref
  1, https://www.jianshu.com/p/d42d330bfead
    简化git subtree命令

    我们已经知道了git subtree 的命令的基本用法，但是上述几个命令还是显得有点复杂，特别是子仓库的源仓库地址，特别不方便记忆。
    这里我们把子仓库的地址作为一个remote，方便记忆：
    git remote add -f libpng https://github.com/test/libpng.git

    然后可以这样来使用git subtree命令：
    git subtree add --prefix=sub/libpng libpng master --squash
    git subtree pull --prefix=sub/libpng libpng master --squash
    git subtree push --prefix=sub/libpng libpng master


    作者：好好编程
    链接：https://www.jianshu.com/p/d42d330bfead
    来源：简书
    著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
