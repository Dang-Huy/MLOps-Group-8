# ═══════════════════════════════════════════════════════════════════════════════
# MLOps Group 8 - Kubernetes Deployment Makefile
# 
# Quick commands for deploying Credit Score API to a local kind cluster
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all cluster deploy status clean logs web help describe events shell

# Configuration variables
CLUSTER_NAME      = mlops-cluster
NAMESPACE         = credit-score
IMAGE_NAME        = ruoc188/mlops-group8:v1.0
MODEL_IMAGE_NAME  = ruoc188/ml-models:latest

# Kubernetes manifests directory
K8S_DIR           = deployment/k8s

# Default target: show help
help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║   MLOps Group 8 - Credit Score API - Kubernetes Deployment    ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 Main commands:"
	@echo "  make all          Full deployment (cluster + deploy + status)"
	@echo "  make cluster      Create kind cluster"
	@echo "  make deploy       Deploy to existing cluster"
	@echo "  make status       Show deployment status"
	@echo "  make logs         Stream API logs"
	@echo "  make web          Port-forward to localhost:8000"
	@echo "  make clean        Delete cluster"
	@echo ""
	@echo "🔧 Utility commands:"
	@echo "  make describe     Show API deployment details"
	@echo "  make events       Show recent cluster events"
	@echo "  make shell        Open shell in API pod"
	@echo ""

# Full workflow: create cluster → deploy → show status
all: cluster deploy status

# Create kind cluster with configuration
cluster:
	@echo "🔨 Creating kind cluster: $(CLUSTER_NAME)"
	kind create cluster --name $(CLUSTER_NAME) --config $(K8S_DIR)/kind-config.yaml || \
	  (echo "⚠️  Cluster already exists"; exit 0)
	@echo "✅ Cluster created"

# Deploy all manifests
deploy: cluster
	@echo "📦 Loading Docker images..."
	-kind load docker-image $(IMAGE_NAME) --name $(CLUSTER_NAME) 2>/dev/null
	-kind load docker-image $(MODEL_IMAGE_NAME) --name $(CLUSTER_NAME) 2>/dev/null
	@echo "📝 Applying Kubernetes manifests..."
	kubectl apply -f $(K8S_DIR)/namespace.yaml
	kubectl apply -f $(K8S_DIR)/configmap.yaml
	kubectl apply -f $(K8S_DIR)/pvc.yaml
	kubectl apply -f $(K8S_DIR)/api-deployment.yaml
	kubectl apply -f $(K8S_DIR)/api-service.yaml
	kubectl apply -f $(K8S_DIR)/hpa.yaml
	@echo "⏳ Waiting for pods to be ready..."
	kubectl -n $(NAMESPACE) rollout status deploy/credit-score-api --timeout=180s

# Show deployment status
status:
	@echo ""
	@echo "📊 Kubernetes Resources in namespace: $(NAMESPACE)"
	@echo "════════════════════════════════════════════════════════"
	kubectl -n $(NAMESPACE) get pods -o wide
	@echo ""
	kubectl -n $(NAMESPACE) get svc
	@echo ""
	kubectl -n $(NAMESPACE) get hpa
	@echo ""
	@echo "💡 Run: make web"

# Stream logs
logs:
	@echo "📋 Streaming logs (Ctrl+C to stop)..."
	kubectl -n $(NAMESPACE) logs -f deploy/credit-score-api --all-containers=true

# Port-forward to localhost
web:
	@echo "🌐 Forwarding to http://localhost:8000"
	kubectl -n $(NAMESPACE) port-forward svc/credit-score-api 8000:8000

# Delete cluster
clean:
	@echo "🗑️  Deleting cluster: $(CLUSTER_NAME)"
	kind delete cluster --name $(CLUSTER_NAME)
	@echo "✅ Done"

# Utility: Show deployment details
describe:
	kubectl -n $(NAMESPACE) describe deploy/credit-score-api

# Utility: Show recent events
events:
	kubectl -n $(NAMESPACE) get events --sort-by='.lastTimestamp' | tail -20

# Utility: Shell into pod
shell:
	@POD=$$(kubectl -n $(NAMESPACE) get pods -l app=credit-score-api -o jsonpath='{.items[0].metadata.name}'); \
	kubectl -n $(NAMESPACE) exec -it $$POD -- /bin/bash