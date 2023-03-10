for GNN_TYPE in "CGConv" "NNConv" "TransformerConv" "GINEConv" "GATConv" "GCNConv"; do
  echo ./runCommandOnGPUMemThreshold.sh "python -m train.zinc model.gnn_type ${GNN_TYPE} best params" 35000
  echo ./runCommandOnGPUMemThreshold.sh "python -m train.molhiv model.gnn_type ${GNN_TYPE} best params" 35000
  echo ./runCommandOnGPUMemThreshold.sh "python -m train.proximity task 10 model.gnn_type ${GNN_TYPE} best params" 35000
done

