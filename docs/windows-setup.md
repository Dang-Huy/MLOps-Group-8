# Windows Setup Guide with Chocolatey

This guide walks you through setting up the entire MLOps project on **Windows** using **Chocolatey** package manager and deploying to Kubernetes using **kind**.

---

## 📋 Prerequisites

- **Windows 10/11** (Pro or Enterprise for Hyper-V)
- **Administrator access** to install packages
- **PowerShell 5.0+** (usually included on Windows)
- **Internet connection** to download packages from Chocolatey

---

## 🍫 Step 1: Install Chocolatey

### 1.1 Open PowerShell as Administrator

1. Press `Win + X` on your keyboard
2. Select "Windows PowerShell (Admin)" or "Terminal (Admin)"
3. Click "Yes" when prompted

### 1.2 Run the installation command

Copy and paste this command into the PowerShell window:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

This command:

- Allows PowerShell to execute scripts temporarily
- Downloads and runs the Chocolatey installer

### 1.3 Verify installation

Close the PowerShell window and open a new one as Administrator. Run:

```powershell
choco --version
```

You should see a version number (e.g., `v2.2.2`).

---

## 🐳 Step 2: Install Docker Desktop

### 2.1 Install via Chocolatey

```powershell
choco install docker-desktop -y
```

### 2.2 Start Docker Desktop

1. Press `Win` to open Start menu
2. Search for "Docker Desktop"
3. Click to open
4. Wait for the Docker daemon to start (watch the system tray icon in the bottom-right)

### 2.3 Verify Docker is running

```powershell
docker ps
```

You should see a list of containers (may be empty initially). If this fails, Docker is not running.

---

## ☸️ Step 3: Install Kubernetes Tools

### 3.1 Install kubectl

```powershell
choco install kubectl -y
```

Verify:

```powershell
kubectl version --client
```

### 3.2 Install kind (local Kubernetes cluster)

```powershell
choco install kind -y
```

Verify:

```powershell
kind --version
```

### 3.3 Install Make (for Makefile commands)

```powershell
choco install make -y
```

Verify:

```powershell
make --version
```

### 3.4 (Optional) Install Helm (for monitoring stack)

```powershell
choco install kubernetes-helm -y
```

Verify:

```powershell
helm version
```

---

## 🔧 Step 4: Verify All Tools

Run this in PowerShell to check all installations:

```powershell
Write-Host "Docker:" -ForegroundColor Green
docker --version

Write-Host "Docker Compose:" -ForegroundColor Green
docker compose version

Write-Host "kubectl:" -ForegroundColor Green
kubectl version --client

Write-Host "kind:" -ForegroundColor Green
kind --version

Write-Host "make:" -ForegroundColor Green
make --version
```

Expected output:

```bash
Docker: Docker version 24.0.0
Docker Compose: Docker Compose version v2.x.x
kubectl: Client Version: v1.x.x
kind: kind version 0.20.0
make: GNU Make 4.x.x
```

---

## 📁 Step 5: Clone and Prepare Repository

### 5.1 Clone the project

```powershell
cd D:\code\Python\DSEB_ex  # Or your preferred path
git clone https://github.com/your-org/MLOps-Group-8.git
cd MLOps-Group-8
```

### 5.2 Install Python dependencies

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 5.3 Set up MLflow (for local development)

In a separate PowerShell window:

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Keep this window open. You'll see:

```bash
[2024-01-15 10:00:00 +0000] [1234] [INFO] Listening at: http://127.0.0.1:5000
```

### 5.4 Run the training pipeline

In another PowerShell window (with MLflow running):

```powershell
cd MLOps-Group-8
python -m src.pipelines.training_pipeline
```

This trains the model and creates artifacts.

---

## 🚀 Step 6: Deploy to Kubernetes

### 6.1 Quick deployment with Make

```powershell
cd MLOps-Group-8
make all
```

This single command:

1. Creates a kind cluster (`mlops-cluster`)
2. Loads Docker images
3. Applies all Kubernetes manifests
4. Waits for Pods to be ready
5. Shows deployment status

Expected output:

```bash
🔨 Creating kind cluster: mlops-cluster
✅ Cluster created
📦 Loading Docker images...
📝 Applying Kubernetes manifests...
⏳ Waiting for pods to be ready...
Waiting for deployment "credit-score-api" rollout to finish: 2 of 2 updated replicas are available

📊 Kubernetes Resources in namespace: credit-score
NAME                            READY   STATUS    RESTARTS   AGE
pod/credit-score-api-abc123     2/2     Running   0          5s
pod/credit-score-api-def456     2/2     Running   0          5s
...
```

### 6.2 Port-forward to localhost

In a **new PowerShell window**:

```powershell
make web
```

You'll see:

```bash
🌐 Forwarding to http://localhost:8000
Forwarding from 127.0.0.1:8000 -> 8000
```

Keep this window open.

### 6.3 Test the API

In another PowerShell window:

```powershell
# Health check
curl http://localhost:8000/health

# Swagger UI (open in browser)
start http://localhost:8000/docs

# Web UI (open in browser)
start http://localhost:8000/ui/

# Model info
curl http://localhost:8000/model-info
```

---

## 📊 Step 7: Monitor and Debug

### Check deployment status

```powershell
make status
```

### View API logs

```powershell
make logs
```

Press `Ctrl+C` to stop streaming.

### Describe deployment

```powershell
make describe
```

### Open shell in API pod

```powershell
make shell
```

---

## 🧹 Step 8: Cleanup

### Stop port-forward

In the port-forward terminal, press `Ctrl+C`.

### Stop MLflow server

In the MLflow terminal, press `Ctrl+C`.

### Delete Kubernetes cluster

```powershell
make clean
```

This deletes the `mlops-cluster` and all resources.

---

## 🔗 Makefile Commands Reference

| Command | Description |
| --------- | ------------- |
| `make all` | Full setup (cluster → deploy → status) |
| `make cluster` | Create kind cluster only |
| `make deploy` | Deploy to existing cluster |
| `make status` | Show deployment status |
| `make logs` | Stream API logs |
| `make web` | Port-forward to localhost:8000 |
| `make clean` | Delete cluster |
| `make describe` | Show deployment details |
| `make events` | Show cluster events |
| `make shell` | Open shell in pod |

---

## 🛠️ Troubleshooting

### Issue: "Docker daemon is not running"

**Solution:**

1. Open Docker Desktop from Start menu
2. Wait 30 seconds for daemon to initialize
3. Run `docker ps` to verify

### Issue: "kind: command not found"

**Solution:**

```powershell
choco install kind -y
```

Close and reopen PowerShell.

### Issue: "kubectl: command not found"

**Solution:**

```powershell
choco install kubectl -y
```

Close and reopen PowerShell.

### Issue: "make: command not found"

**Solution:**

```powershell
choco install make -y
```

Close and reopen PowerShell.

### Issue: "Pods not starting" or stuck in "Pending"

**Solution:**

```powershell
make logs
make describe
kubectl -n credit-score get events --sort-by='.lastTimestamp'
```

Check for image pull errors or resource issues.

### Issue: "Port 8000 already in use"

**Solution:**

```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with the number from above)
taskkill /PID <PID> /F

# Or use a different port
kubectl -n credit-score port-forward svc/credit-score-api 8001:8000
```

### Issue: MLflow cannot connect from K8s

**Solution:**

Edit `deployment/k8s/configmap.yaml` and change:

```yaml
mlflow_tracking_uri: "http://host.docker.internal:5000"
```

This connects to MLflow running on your Windows host.

---

## 📚 Next Steps

- Read [README.md](../README.md) for full project overview
- See [architecture.md](./architecture.md) for system design
- Check [runbook.md](./runbook.md) for operational guides
- Review [api_spec.md](./api_spec.md) for API endpoint documentation

---

## ❓ Questions?

- Check logs: `make logs`
- Show events: `make events`
- View deployment: `make describe`
- Shell into pod: `make shell`
