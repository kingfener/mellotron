# ubuntu18-env-ins
  记录 ubunu18的 装机常规操作。

# 好的文章 ref: 
  https://pabebezz.github.io/article/e0fdde43/









# 硬盘挂载

  ## 支持 exfat格式
    sudo add-apt-repository universe
    sudo apt install exfat-fuse exfat-utils
  ## 初步挂载
    sudo mount /dev/sdc2 /media/U1T
    sudo mount /dev/sda1 /media/3tk
    sudo mount /dev/sdb2 /media/3B
    # 热风： https://askubuntu.com/questions/11840/how-do-i-use-chmod-on-an-ntfs-or-fat32-partition/956072#956072
      sudo mount -o rw,users,umask=000,exec /dev/sdc2 /media/U1T
      sudo mount -o rw,users,umask=000,exec /dev/sda1 /media/3tk
      sudo mount -o rw,users,umask=000,exec /dev/sdb2 /media/3B
    
  ## 自动挂载
    #查看 UUID
    #https://blog.csdn.net/buxiaoxindasuile/article/details/49612867
    #https://www.linuxprobe.com/linux-uuid.html
    sudo blkid
    sudo nano /etc/fstab
      UUID=5F3B-D1FF /media/3B exfat auto,user,rw 0 0
      UUID=96BF-03CD /media/3tk exfat auto,user,rw 0 0
      UUID=5D7A-2DBF /media/U1T exfat auto,user,rw 0 0

  ## 备注： 
    当前 typora 仍然无法在 U1T硬盘中直接把修改文件。

# 讲当前用户加入 sudo组
    #sudo su
    sudo nano /etc/sudoers
    add : king	ALL=(ALL::ALL) ALL
     
  ## 把用户user的用户组改为staff吗？
    命令：
    usermod -g staff user
    使用root 执行上述命令










