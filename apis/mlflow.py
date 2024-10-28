import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGExperimentTracker:
    def __init__(self, experiment_name: str):
        """Initialize the experiment tracker with local storage."""
        self.experiment_name = experiment_name
        self.base_dir = Path(f"experiments/{experiment_name}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        
        # Initialize metrics storage with all possible metrics
        self.metrics_file = self.base_dir / "metrics.csv"
        if not self.metrics_file.exists():
            pd.DataFrame(columns=[
                'run_id', 
                'timestamp', 
                'retriever_accuracy',
                'validation_accuracy',
                'relevance_score',
                'groundedness_score',
                'avg_response_time',
                'avg_response_length', 
                'avg_query_length',
                'query_count',
                'model_config',
                'experiment_name',
                'experiment_type'
            ]).to_csv(self.metrics_file, index=False)
        
        logger.info(f"Initialized experiment tracker for {experiment_name}")

    def update_experiment_metrics(self, 
                                experiment_name: str,
                                experiment_type: str = None,
                                model_config: dict = None,
                                **metrics):
        """
        Update metrics for a specific experiment.
        
        Args:
            experiment_name (str): Name of the experiment
            experiment_type (str): Type of experiment (e.g., 'BM25', 'Dense')
            model_config (dict): Model configuration
            **metrics: Arbitrary metric key-value pairs
        """
        try:
            timestamp = datetime.now()
            run_id = f"run_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Create new metrics row
            new_metrics = {
                'run_id': run_id,
                'timestamp': timestamp.isoformat(),
                'experiment_name': experiment_name,
                'experiment_type': experiment_type,
                'model_config': str(model_config) if model_config else None
            }
            
            # Add all provided metrics
            new_metrics.update(metrics)
            
            # Remove None values
            new_metrics = {k: v for k, v in new_metrics.items() if v is not None}
            
            # Read existing metrics
            metrics_df = pd.read_csv(self.metrics_file)
            
            # Add new row
            metrics_df = pd.concat([metrics_df, pd.DataFrame([new_metrics])], ignore_index=True)
            
            # Save updated metrics
            metrics_df.to_csv(self.metrics_file, index=False)
            
            logger.info(f"Successfully updated metrics for experiment: {experiment_name}")
            return run_id
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            raise

    def log_query_metrics(self,
                         experiment_name: str,
                         query: str,
                         response: str,
                         relevance_score: float,
                         groundedness_score: float,
                         response_time: float):
        """Log metrics for individual queries."""
        try:
            # Get existing metrics for the experiment
            metrics_df = self.get_experiment_metrics(experiment_name)
            
            # Calculate new averages
            current_metrics = {
                'relevance_score': relevance_score,
                'groundedness_score': groundedness_score,
                'avg_response_time': response_time,
                'avg_response_length': len(response),
                'avg_query_length': len(query),
                'query_count': 1
            }
            
            # Update experiment metrics
            self.update_experiment_metrics(
                experiment_name=experiment_name,
                **current_metrics
            )
            
            logger.info(f"Successfully logged query metrics for experiment: {experiment_name}")
            
        except Exception as e:
            logger.error(f"Error logging query metrics: {str(e)}")
            raise

    def get_experiment_metrics(self, experiment_name: str = None, experiment_type: str = None):
        """Get metrics filtered by experiment name and/or type."""
        try:
            metrics_df = pd.read_csv(self.metrics_file)
            
            if experiment_name:
                metrics_df = metrics_df[metrics_df['experiment_name'] == experiment_name]
            
            if experiment_type:
                metrics_df = metrics_df[metrics_df['experiment_type'] == experiment_type]
                
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error reading metrics: {str(e)}")
            return pd.DataFrame()

    def get_latest_metrics(self, experiment_name: str = None, experiment_type: str = None):
        """Get the latest metrics filtered by experiment name and/or type."""
        try:
            metrics_df = self.get_experiment_metrics(experiment_name, experiment_type)
            return metrics_df.iloc[-1] if not metrics_df.empty else None
        except Exception as e:
            logger.error(f"Error reading latest metrics: {str(e)}")
            return None

    def get_metrics_summary(self, experiment_name: str = None, experiment_type: str = None):
        """Get summary statistics for metrics."""
        try:
            metrics_df = self.get_experiment_metrics(experiment_name, experiment_type)
            
            if metrics_df.empty:
                return None
            
            numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns
            summary = metrics_df[numeric_columns].agg(['mean', 'min', 'max', 'std']).round(4)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating metrics summary: {str(e)}")
            return None

def main():
    # Example usage
    tracker = RAGExperimentTracker("RAG_Experiments")
    
    # Example 1: Update metrics for BM25 experiment
    # Dense Retrieval Fixed Text Experiment
    tracker.update_experiment_metrics(
        experiment_name="Dense_Fixed_Text_Experiment",
        experiment_type="Dense_Fixed",
        model_config="",
        retriever_accuracy=0.68,
        validation_accuracy=0.64,
        relevance_score=0.72,
        groundedness_score=0.69,
        avg_response_time=0.45
    )
    
    # Sparse Retrieval Experiment
    tracker.update_experiment_metrics(
        experiment_name="Sparse_Retrieval_Experiment", 
        experiment_type="Sparse",
        model_config="KeyWord Based Search , BM 25",
        retriever_accuracy=0.58,
        validation_accuracy=0.59,
        relevance_score=0.65,
        groundedness_score=0.45,
        avg_response_time=0.43
    )
    
    # Hybrid Retrieval Experiment
    tracker.update_experiment_metrics(
        experiment_name="Hybrid_Retrieval_Experiment",
        experiment_type="Hybrid",
        model_config="Embedding Based Search",
        retriever_accuracy=0.74,
        validation_accuracy=0.77,
        relevance_score=0.76,
        groundedness_score=0.83,
        avg_response_time=0.43
    )
    
    # Dense Retrieval Semantic Experiment
    tracker.update_experiment_metrics(
        experiment_name="Dense_Semantic_Experiment",
        experiment_type="Dense_Semantic", 
        model_config="Embedding + KEywords Based Search",
        retriever_accuracy=0.76,
        validation_accuracy=0.73,
        relevance_score=0.79,
        groundedness_score=0.87,
        avg_response_time=0.44
    )
    
    # Dense Retrieval Contextual Experiment
    tracker.update_experiment_metrics(
        experiment_name="Dense_Contextual_Experiment",
        experiment_type="Dense_Contextual",
        model_config="CHunks were contextualized and Embedding Based Search", 
        retriever_accuracy=0.81,
        validation_accuracy=0.78,
        relevance_score=0.85,
        groundedness_score=0.83,
        avg_response_time=0.43
    )

    tracker.update_experiment_metrics(
        experiment_name="Metadata_Filtering_Dense_Contextual_Experiment",
        experiment_type="Metadat_Contextual_dense",
        model_config="Metadata FIltering APplied Based ON Category,CHunks were contextualized and Embedding Based Search ", 
        retriever_accuracy=0.83,
        validation_accuracy=0.79,
        relevance_score=0.86,
        groundedness_score=0.89,
        avg_response_time=0.43
    )
        
    # Get metrics summary
    print("\nMetrics Summary for all experiments:")
    print(tracker.get_metrics_summary())
    
    # Get metrics for specific experiment type
    print("\nMetrics for BM25 experiments:")
    print(tracker.get_experiment_metrics(experiment_type="BM25"))
    
    # Get latest metrics for specific experiment
    print("\nLatest metrics for Dense Retrieval Experiment:")
    print(tracker.get_latest_metrics("Dense_Retrieval_Experiment"))

if __name__ == "__main__":
    main()
