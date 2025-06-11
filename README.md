# trust_socket
## 所需环境
- Python 版本：Python 3.8 或更高版本
## 运行说明
### 1. 设置环境变量
在运行服务端和客户端之前，需要设置环境变量 TRUST_ID，用于身份验证，服务端和客户端必须使用相同的值。  
在 Linux 或 macOS 系统中，Bash		export TRUST_ID="your_trust_id"  
在 Windows 系统中，Cmd		set TRUST_ID=your_trust_id
### 2.启动服务端
在项目根目录下，运行以下命令启动服务端：python server.py，服务端将监听 “127.0.0.1:8888”地址和端口。
### 3.启动客户端
在另一个终端窗口中，运行以下命令启动客户端：python client.py，客户端将尝试连接到服务端并发送身份验证信息。如果身份验证成功，客户端将发送一条消息到服务端，并接收服务端的回显消息。
### 4.测试
test.py测试客户端和服务端的通信逻辑。运行以下命令执行测试：pytest test.py，测试用例将启动服务端和客户端，模拟身份匹配和不匹配的情况，验证通信逻辑是否正常。
## 基础题目（socket）运行效果展示
### 1. 运行server.py和client.py
![image](https://github.com/user-attachments/assets/14bd6ce4-2220-4388-ad6a-c2c074312818)
### 2. 运行test.py测试

## 附加题目（RPC）运行效果展示
### 1. 运行server_RPC.py和client_RPC.py
![image](https://github.com/user-attachments/assets/fe7f2590-2f80-4725-b569-ecee1ebda7d8)
![image](https://github.com/user-attachments/assets/7be73d39-e1b9-4c6b-a411-c3ecb40c5742)
### 2. 运行test_RPC.py测试
![image](https://github.com/user-attachments/assets/54f09591-c643-49d7-a284-41aaeb1688f1)
![image](https://github.com/user-attachments/assets/971ed52c-a9bc-4c05-9cd5-4faae6ec1dc8)

