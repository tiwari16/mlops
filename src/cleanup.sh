#!/bin/bash

# Define variables
RESOURCE_GROUP="diabetes-dev-rg"
WORKSPACE="diabetes-ws"
COMPUTE_NAME="diabetescompute"
DATA_ASSET_NAME="diabetes-dev-folder"

# Make sure you're logged in
az account show > /dev/null 2>&1 || az login

echo "🧹 Deleting compute instance (if exists)..."
az ml compute delete --name $COMPUTE_NAME \
  --workspace-name $WORKSPACE \
  --resource-group $RESOURCE_GROUP \
  --yes || echo "⚠️  Compute instance not found or already deleted."

echo "🧹 Deleting data asset (if exists)..."
az ml data delete --name $DATA_ASSET_NAME \
  --workspace-name $WORKSPACE \
  --resource-group $RESOURCE_GROUP \
  --yes || echo "⚠️  Data asset not found or already deleted."

echo "🧹 Deleting workspace (if exists)..."
az ml workspace delete --name $WORKSPACE \
  --resource-group $RESOURCE_GROUP \
  --yes || echo "⚠️  Workspace not found or already deleted."

echo "🧹 Deleting resource group (if exists)..."
az group delete --name $RESOURCE_GROUP --yes --no-wait || echo "⚠️  Resource group not found or already deleted."

echo "✅ All Azure resources scheduled for deletion."
echo "Cleanup completed successfully. You can now re-run setup.sh to recreate the environment."