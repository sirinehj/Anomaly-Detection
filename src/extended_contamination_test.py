# Importing required packages
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from yaml import CLoader as Loader, load
from ml_pipeline.utils import read_data_csv, read_config, find_contamination
from ml_pipeline.preprocessing import handle_null_values
from ml_pipeline.model import train_IF, train_lof, predict_scores, anomaly_scores, save_model

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def evaluate_contamination_extensive(X, y, contamination_values=None):
    """
    Extended contamination testing with more values and detailed analysis
    """
    if contamination_values is None:
        contamination_values = [
            0.0001, 0.0005, 0.001, 0.0018, 0.003, 0.005, 
            0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
            0.06, 0.07, 0.08, 0.09, 0.1
        ]
    
    results = {}
    detailed_results = []
    
    print("Testing contamination values...")
    print("=" * 80)
    print(f"{'Contamination':<15} {'Recall':<10} {'Precision':<12} {'F1-Score':<10} {'Frauds Caught':<15} {'Total Flagged':<15}")
    print("=" * 80)
    
    for cont in contamination_values:
        # Train model with current contamination
        clf = train_IF(X, contamination=cont)
        predictions = clf.predict(X)
        
        # Convert predictions to binary format
        y_pred = (predictions == -1).astype(int)
        
        # Calculate metrics
        tp = ((y == 1) & (y_pred == 1)).sum()
        fp = ((y == 0) & (y_pred == 1)).sum()
        fn = ((y == 1) & (y_pred == 0)).sum()
        tn = ((y == 0) & (y_pred == 0)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        total_flagged = y_pred.sum()
        
        # Store results
        results[cont] = {
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'frauds_caught': tp,
            'total_flagged': total_flagged,
            'false_positives': fp
        }
        
        detailed_results.append({
            'contamination': cont,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'frauds_caught': tp,
            'total_flagged': total_flagged
        })
        
        print(f"{cont:<15} {recall:<10.3f} {precision:<12.3f} {f1:<10.3f} {tp:<15} {total_flagged:<15}")
    
    return results, detailed_results

def find_optimal_contamination(results):
    """
    Find optimal contamination based on different criteria
    """
    print("\n" + "=" * 60)
    print("OPTIMAL CONTAMINATION ANALYSIS")
    print("=" * 60)
    
    # Best recall
    best_recall_cont = max(results.keys(), key=lambda x: results[x]['recall'])
    best_recall_value = results[best_recall_cont]['recall']
    
    # Best precision
    best_precision_cont = max(results.keys(), key=lambda x: results[x]['precision'])
    best_precision_value = results[best_precision_cont]['precision']
    
    # Best F1-score
    best_f1_cont = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_f1_value = results[best_f1_cont]['f1_score']
    
    # Best balance (high recall with reasonable precision > 0.01)
    balanced_candidates = {k: v for k, v in results.items() if v['precision'] > 0.01}
    if balanced_candidates:
        best_balanced_cont = max(balanced_candidates.keys(), key=lambda x: balanced_candidates[x]['recall'])
        best_balanced_recall = balanced_candidates[best_balanced_cont]['recall']
        best_balanced_precision = balanced_candidates[best_balanced_cont]['precision']
    else:
        best_balanced_cont = best_f1_cont
        best_balanced_recall = results[best_balanced_cont]['recall']
        best_balanced_precision = results[best_balanced_cont]['precision']
    
    print(f"🎯 Best Recall: {best_recall_cont} (Recall: {best_recall_value:.3f})")
    print(f"🎯 Best Precision: {best_precision_cont} (Precision: {best_precision_value:.3f})")
    print(f"🎯 Best F1-Score: {best_f1_cont} (F1: {best_f1_value:.3f})")
    print(f"🎯 Best Balanced: {best_balanced_cont} (Recall: {best_balanced_recall:.3f}, Precision: {best_balanced_precision:.3f})")
    
    return best_recall_cont, best_precision_cont, best_f1_cont, best_balanced_cont

def plot_contamination_results(detailed_results):
    """
    Plot contamination results for visualization
    """
    df = pd.DataFrame(detailed_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Recall vs Contamination
    axes[0,0].plot(df['contamination'], df['recall'], 'b-o', linewidth=2, markersize=6)
    axes[0,0].set_title('Recall vs Contamination', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Contamination')
    axes[0,0].set_ylabel('Recall')
    axes[0,0].grid(True, alpha=0.3)
    
    # Precision vs Contamination
    axes[0,1].plot(df['contamination'], df['precision'], 'r-o', linewidth=2, markersize=6)
    axes[0,1].set_title('Precision vs Contamination', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Contamination')
    axes[0,1].set_ylabel('Precision')
    axes[0,1].grid(True, alpha=0.3)
    
    # F1-Score vs Contamination
    axes[1,0].plot(df['contamination'], df['f1_score'], 'g-o', linewidth=2, markersize=6)
    axes[1,0].set_title('F1-Score vs Contamination', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Contamination')
    axes[1,0].set_ylabel('F1-Score')
    axes[1,0].grid(True, alpha=0.3)
    
    # Frauds Caught vs Contamination
    axes[1,1].plot(df['contamination'], df['frauds_caught'], 'm-o', linewidth=2, markersize=6)
    axes[1,1].set_title('Frauds Caught vs Contamination', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Contamination')
    axes[1,1].set_ylabel('Number of Frauds Caught')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('contamination_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_best_contamination_with_ensemble(X, y, best_contaminations):
    """
    Test the best contamination values with ensemble approach
    """
    print("\n" + "=" * 60)
    print("TESTING BEST CONTAMINATIONS WITH ENSEMBLE")
    print("=" * 60)
    
    for name, cont_value in best_contaminations.items():
        print(f"\n--- Testing {name} (contamination={cont_value}) ---")
        
        # Train IF with best contamination
        clf_if = train_IF(X, contamination=cont_value)
        if_predictions = clf_if.predict(X)
        if_binary = (if_predictions == -1).astype(int)
        
        # Train LOF with same contamination
        lof = train_lof(X, contamination=cont_value, n_neighbors=50)
        lof_scores = anomaly_scores(lof)
        
        # Create ensemble
        lof_threshold = np.percentile(lof_scores, (1-cont_value)*100)
        lof_predictions = (lof_scores < lof_threshold).astype(int)
        
        # Different ensemble strategies
        ensemble_union = (if_binary | lof_predictions).astype(int)
        ensemble_intersection = (if_binary & lof_predictions).astype(int)
        
        # Weighted ensemble
        if_scores = clf_if.decision_function(X)
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
        
        ensemble_scores = 0.6 * (1 - if_scores_norm) + 0.4 * (1 - lof_scores_norm)
        ensemble_threshold = np.percentile(ensemble_scores, (1-cont_value)*100)
        ensemble_weighted = (ensemble_scores > ensemble_threshold).astype(int)
        
        # Evaluate all methods
        methods = {
            'IF Only': if_binary,
            'LOF Only': lof_predictions,
            'Union': ensemble_union,
            'Intersection': ensemble_intersection,
            'Weighted': ensemble_weighted
        }
        
        print(f"{'Method':<15} {'Frauds':<10} {'Flagged':<10} {'Recall':<10} {'Precision':<10}")
        print("-" * 55)
        
        for method_name, predictions in methods.items():
            tp = ((y == 1) & (predictions == 1)).sum()
            total_flagged = predictions.sum()
            
            recall = tp / 253  # Total frauds = 253
            precision = tp / total_flagged if total_flagged > 0 else 0
            
            print(f"{method_name:<15} {tp:<10} {total_flagged:<10} {recall:<10.3f} {precision:<10.3f}")

# Main execution
if __name__ == "__main__":
    # Reading config file
    config = read_config("input/config.yaml")
    
    # Reading the data
    transaction_data = read_data_csv(config['data_path'])
    print(f"Dataset loaded: {transaction_data.shape}")
    
    # Handling missing values
    transaction_data = handle_null_values(transaction_data)
    
    # Prepare features and target
    X = transaction_data.drop('Class', axis=1)
    y = transaction_data['Class']
    
    print(f"Total transactions: {len(y)}")
    print(f"Fraud transactions: {(y == 1).sum()}")
    print(f"Normal transactions: {(y == 0).sum()}")
    
    # Extended contamination testing
    print("\n" + "=" * 60)
    print("EXTENDED CONTAMINATION TESTING")
    print("=" * 60)
    
    results, detailed_results = evaluate_contamination_extensive(X, y)
    
    # Find optimal contamination values
    best_recall_cont, best_precision_cont, best_f1_cont, best_balanced_cont = find_optimal_contamination(results)
    
    # Plot results
    plot_contamination_results(detailed_results)
    
    # Test best contamination values with ensemble
    best_contaminations = {
        'Best Recall': best_recall_cont,
        'Best Precision': best_precision_cont,
        'Best F1-Score': best_f1_cont,
        'Best Balanced': best_balanced_cont
    }
    
    test_best_contamination_with_ensemble(X, y, best_contaminations)
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print(f"• For maximum fraud detection: Use contamination = {best_recall_cont}")
    print(f"• For balanced performance: Use contamination = {best_balanced_cont}")
    print(f"• For minimum false alarms: Use contamination = {best_precision_cont}")
    print("• Consider ensemble approaches for better overall performance")
    print("=" * 60)