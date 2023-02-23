for GNN_TYPE in "CGConv" "NNConv" "TransformerConv" "GINEConv" "GATConv" "GCNConv"; do
  echo "python -m train.A model.gnn_type ${GNN_TYPE} best params"
  echo "python -m train.B model.gnn_type ${GNN_TYPE} best params"
  echo "python -m train.C model.gnn_type ${GNN_TYPE} best params"
done

