# 自助测试

> 适用范围：采用容器内授权方案

* 获取镜像，**[ [ubuntu18.04](https://hub.docker.com/layers/msbrm/fat_gpu/2023ubuntu18.04/images/sha256-24897635dc807a07dcfd05efe36c1627391ee53797e80161de1c16f51317f234?context=repo), [centos7](https://hub.docker.com/layers/msbrm/fat_gpu/2023centos7/images/sha256-04aa1fe30aab1f349688fdcab5642803be7968d63ff14b2c44b47fb86b29e92f?context=repo) ]**

    ```bash
    docker pull msbrm/fat_gpu:2023ubuntu18.04
    ```

    ```bash
    docker pull msbrm/fat_gpu:2023centos7
    ```

* 启动容器与测试服务

    ```bash
    # 1. 启动容器
    # <TAR_DIR> 程序包所在目录
    # <TMP_DIR> 文件交互目录
    # <PY_FILE> 测试脚本名, 可选[face_1n1.py, cluster.py]
    # <IMAGE_ID> 镜像ID
    docker run -idt --gpus all -v <TAR_DIR>:/workspace/tars/ -v <TMP_DIR>:/workspace/projects/ --privileged=true --ipc=host -p 8089:8089 -e EVAL_SHELL=/workspace/container_script/<PY_FILE> -e CEPING_BASE_DIR=/workspace/ <IMAGE_ID>

    # 例如: docker run -idt --gpus all -v /root/test/package/:/workspace/tars/ -v /root/test/tmp/:/workspace/projects/ --privileged=true --ipc=host -p 8089:8089 -e EVAL_SHELL=/workspace/container_script/face_1n1.py -e CEPING_BASE_DIR=/workspace/ 115ce227e4c2

    # 2. 启动容器内测试服务
    # <CONTAINER_ID> 容器ID
    docker exec -it <CONTAINER_ID> /bin/bash -c "/workspace/docker-ceping1/my.sh start"
    ```

* 通过 http 接口调用的方式进行程序包功能性验证
    例如使用 [postman](https://www.postman.com) 调用接口
