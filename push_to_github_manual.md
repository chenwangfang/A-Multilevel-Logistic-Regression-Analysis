# GitHub仓库更新手动指南

## 准备工作

### 1. 检查Git安装
```bash
git --version
```
如果未安装，请从 https://git-scm.com 下载安装

### 2. 配置GitHub认证

#### 方法A：使用Personal Access Token（推荐）

1. 登录GitHub，进入Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 点击"Generate new token (classic)"
3. 设置名称：如"SPAADIA-repo-update"
4. 选择权限：至少勾选`repo`（全部仓库权限）
5. 生成并复制token（只显示一次！）

#### 方法B：使用SSH密钥

```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 复制公钥内容
cat ~/.ssh/id_ed25519.pub

# 添加到GitHub: Settings → SSH and GPG keys → New SSH key
```

## 推送步骤

### 自动方式：运行批处理脚本

直接双击运行：
```
G:\Project\实证\关联框架\github-repo-20250828_080438\push_to_github.bat
```

### 手动方式：逐步执行

打开命令提示符或PowerShell，执行以下命令：

```bash
# 1. 进入仓库目录
cd G:\Project\实证\关联框架\github-repo-20250828_080438

# 2. 初始化Git仓库
git init

# 3. 添加所有文件
git add .

# 4. 查看状态（可选）
git status

# 5. 提交更改
git commit -m "v2.0: Complete rewrite with enhanced statistical analysis and Applied Linguistics compliance"

# 6. 添加远程仓库
git remote add origin https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis.git

# 7. 推送到GitHub（强制覆盖）
git push -u origin main --force

# 如果main分支不存在，尝试master
git push -u origin master --force
```

## 认证处理

### 使用Personal Access Token

当Git要求输入密码时：
- **用户名**：您的GitHub用户名
- **密码**：粘贴Personal Access Token（不是GitHub密码！）

### 保存认证信息（Windows）

```bash
# 保存凭据，避免重复输入
git config --global credential.helper wincred
```

## 常见问题解决

### 问题1：认证失败
```
remote: Support for password authentication was removed
```
**解决**：必须使用Personal Access Token，不能使用密码

### 问题2：分支名称问题
```
error: src refspec main does not match any
```
**解决**：
```bash
# 查看当前分支
git branch

# 如果是master分支
git push -u origin master --force

# 或重命名为main
git branch -M main
git push -u origin main --force
```

### 问题3：仓库不存在
```
remote: Repository not found
```
**解决**：
1. 确认仓库URL正确
2. 确认您有该仓库的写入权限
3. 如果是私有仓库，确保token有相应权限

### 问题4：网络问题
```bash
# 设置代理（如果使用）
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

## 验证更新

推送成功后，访问以下地址验证：
https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis

应该看到：
- ✅ 新的README.md（包含v2.0内容）
- ✅ scripts文件夹（包含8个Python脚本）
- ✅ output文件夹（包含JSON和图表）
- ✅ documentation文件夹（包含中英文手稿）
- ✅ requirements.txt
- ✅ .gitignore

## 后续维护

### 更新文件
```bash
# 修改文件后
git add .
git commit -m "Update: 描述您的更改"
git push origin main
```

### 创建新版本标签
```bash
git tag -a v2.0.0 -m "Version 2.0.0: Full implementation with Applied Linguistics compliance"
git push origin v2.0.0
```

### 创建Release
1. 在GitHub仓库页面点击"Releases"
2. 点击"Create a new release"
3. 选择标签v2.0.0
4. 填写Release说明
5. 上传额外的资源文件（如果需要）

## 需要帮助？

如果遇到其他问题：
1. 检查Git和网络配置
2. 确认GitHub账户权限
3. 查看Git错误信息的详细内容
4. 参考GitHub官方文档：https://docs.github.com