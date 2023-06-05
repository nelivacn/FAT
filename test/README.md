# 自助测试

> 适用范围：采用容器内授权方案的测试项目

* 获取镜像

    ```bash
    docker pull nelivacn/fat_gpu_ubuntu:v1.4
    ```

    ```bash
    docker pull nelivacn/fat_gpu_centos:v2.5
    ```

* 启动容器与测试服务

    ```bash
    # 1. 启动容器
    # <BASE_DIR> 项目目录
    # <PY_FILE> 测试脚本名, 可选[face_1n1.py, cluster.py, vehicle.py]
    # <TASK_ID> 任务ID(程序包文件名下划线分隔最后一个部分, 去除 .tar)
    # <IMAGE_ID> 镜像ID
    docker run -idt --gpus all --privileged=true --ipc=host -p 8089:8089 \
    -v /<BASE_DIR>/FAT/test/log/:/workspace/log/ \
    -v /<BASE_DIR>/FAT/test/tars/:/workspace/tars/ \
    -v /<BASE_DIR>/FAT/test/projects/:/workspace/projects/ \
    -v /<BASE_DIR>/FAT/test/docker-ceping1/:/workspace/docker-ceping1/ \
    -v /<BASE_DIR>/FAT/test/container_script/:/workspace/container_script/ \
    -e NELIVA_EVAL_SHELL=/workspace/container_script/<PY_FILE> \
    -e NELIVA_CEPING_BASE_DIR=/workspace \
    -e NELIVA_TEST_SET_DIR=default \
    -e NELIVA_TASK_ID=<TASK_ID> \
    <IMAGE_ID>

    # 2. 启动容器内测试服务
    # <CONTAINER_ID> 容器ID
    chmod -R 777 /<BASE_DIR>/FAT/test/docker-ceping1/
    docker exec -it <CONTAINER_ID> /bin/bash -c "/workspace/docker-ceping1/my.sh start"
    ```

* 通过 http 接口调用的方式进行程序包功能性验证

    **查看信息**
    调用方式: HTTP GET
    接口地址: http://ip:8089/msg
    输入参数: 无

    **授权**
    调用方式: HTTP GET
    接口地址: http://ip:8089/authorize
    输入参数: 无

    **开始测试**
    调用方式: HTTP GET
    接口地址: http://ip:8089/eval
    输入参数: 无

* 测试流程
    1. 根据使用的镜像操作系统拉取对应的docker镜像
    2. 通过启动容器命令启动容器
    3. 启动容器内的测试服务
    4. 如果需要授权, 请使用宿主机 /\<BASE_DIR\>/FAT/test/projects/\<TASK_ID\>/auth 文件夹下的 \<TASK_ID>_fingerprint.txt 文件进行授权; 如果不需要授权, 请跳过
    5. 如果需要授权, 请将授权文件重命名为 \<TASK_ID>_authorize.txt 并放入宿主机 /\<BASE_DIR\>/FAT/test/projects/\<TASK_ID\>/auth 文件夹下, 然后调用**授权**接口; 如果不需要授权, 请跳过
    6. 调用**开始测试**接口进行测试

* 宿主机 /\<BASE_DIR\>/FAT/test/projects/\<TASK_ID\> 文件夹下有测试日志信息

* 技术相关的问题请在[Issues](https://github.com/nelivacn/FAT/issues)进行提问讨论
