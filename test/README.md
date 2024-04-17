# 自助测试

> 主要测试内容：接口返回值格式、算法稳定性

* 获取该项目到本地工作目录下并赋予文件所需权限
    ```bash
    WORKDIR=/workspace
    mkdir $WORKDIR
    cd $WORKDIR
    git https://github.com/nelivacn/FAT.git
    chmod -R 777 $WORKDIR/FAT/test/selftest/
    ```

* 获取镜像

    ```bash
    docker pull nelivacn/fat_gpu_ubuntu:v1.4
    ```

    ```bash
    docker pull nelivacn/fat_gpu_centos:v2.5
    ```

* 启动测试服务

    ```bash
    cd $WORKDIR/FAT/test/selftest/
    ./my.sh
    ```

* 开始自助测试

    1. 进入测试页面[http://127.0.0.1:8040](http://127.0.0.1:8040)
    2. 输入**镜像名称**与**程序包所在绝对路径**
    3. 点击**初始化**按钮并观察测试日志输出
    4. 如果需要授权请点击**下载**按钮下载指纹文件、**上传**按钮上传授权文件进行授权操作
    5. 等待初始化完成后，点击**测试**开始进行自助测试
    6. 如果当前状态为**测试通过**，请将**测试日志**$WORKDIR/FAT/test/selftest/<task_id>.stlog文件同程序包一同提交
    7. 如果当前状态为**测试失败**，请在日志输出区域中查找失败原因并更正后重新进行自助测试
    
    **该自助测试只涉及接口格式校验以及稳定性测试，不涉及任何性能测试**
* 技术相关的问题请在[Issues](https://github.com/nelivacn/FAT/issues)进行提问讨论
