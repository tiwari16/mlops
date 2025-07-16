#!/bin/bash

# Set names
RESOURCE_GROUP="diabetes-dev-rg"
WORKSPACE="diabetes-ws"
LOCATION="australiaeast"
COMPUTE_NAME="diabetescompute"
DATA_NAME="diabetes-dev-folder"
DATA_PATH="./experimentation/data"  # Local path to folder containing CSV

echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "Creating Azure ML workspace..."
az ml workspace create --name $WORKSPACE --resource-group $RESOURCE_GROUP --location $LOCATION

echo "Creating compute instance..."
az ml compute create \
  --name $COMPUTE_NAME \
  --type ComputeInstance \
  --size Standard_DS11_v2 \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $WORKSPACE

echo "Registering data asset..."
az ml data create \
  --name $DATA_NAME \
  --type uri_folder \
  --path $DATA_PATH \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $WORKSPACE

echo "Azure ML environment ready for job execution."
echo "All resources created successfully."