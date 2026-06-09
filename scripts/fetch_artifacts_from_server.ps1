# 从 AutoDL 服务器同步 score artifact 到本地（供 Streamlit 交互式 ROC）
#
# 用法：修改 $SERVER 后于 PowerShell 执行：
#   .\scripts\fetch_artifacts_from_server.ps1
#
# 或在服务器上先打包再下载：
#   cd /root/autodl-tmp/final_project
#   tar czf /tmp/exp1_artifacts.tar.gz results/runs/*/artifacts/exp1/*.npz
#   scp -P <端口> root@<IP>:/tmp/exp1_artifacts.tar.gz .
#   tar xzf exp1_artifacts.tar.gz -C results/

param(
    [string]$Server = "root@YOUR_SERVER_IP",
    [int]$Port = 22,
    [string]$RemoteRoot = "/root/autodl-tmp/final_project/results",
    [string]$LocalResults = "$PSScriptRoot\..\results"
)

$LocalArtifacts = Join-Path $LocalResults "artifacts\exp1"
New-Item -ItemType Directory -Force -Path $LocalArtifacts | Out-Null

Write-Host "同步 exp1 artifacts: $Server -> $LocalArtifacts"

# 合并各 seed run 目录下的 npz（文件名含 dataset+algorithm，本地按名匹配）
$runs = ssh -p $Port $Server "find $RemoteRoot/runs -path '*/artifacts/exp1/*.npz' 2>/dev/null"
if (-not $runs) {
    Write-Host "未找到远程 artifact；尝试 results/artifacts/exp1/"
    scp -P $Port -r "${Server}:${RemoteRoot}/artifacts/exp1/*" $LocalArtifacts
} else {
    foreach ($remote in ($runs -split "`n")) {
        if ($remote.Trim()) {
            scp -P $Port "${Server}:$remote" $LocalArtifacts
        }
    }
}

Write-Host "完成。请刷新 Streamlit 基准对比页查看交互式 ROC。"
