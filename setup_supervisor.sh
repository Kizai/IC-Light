#!/bin/bash

# 创建日志目录
mkdir -p /root/IC-Light/logs

# 安装supervisor
apt-get update
apt-get install -y supervisor

# 复制配置文件
cp supervisor/ic-light.conf /etc/supervisor/conf.d/

# 创建日志文件
touch /root/IC-Light/logs/main.log
touch /root/IC-Light/logs/api.log

# 重新加载supervisor配置
supervisorctl reread
supervisorctl update

# 启动服务
supervisorctl start ic-light:* 