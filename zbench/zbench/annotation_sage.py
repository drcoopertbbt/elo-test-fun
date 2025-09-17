"""
Modified ZBench annotation pipeline with SageMath Elo calculation engine
This version replaces the internal calculate_elos with our high-precision SageMath service
"""

import asyncio
from pathlib import Path
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm
import random
from shutil import copy2
import subprocess
import json
import tempfile
import os

from zbench.utils import (
    load_jsonl, load_json, save_pydantic_jsonl, save_pydantic_json
)
from zbench.common_types import DatasetPairs, DatasetPair, Dataset, AnnotatedDataset, AnnotatedQueryDocuments, AnnotatedDocument, DatasetPairScoredPairs, DatasetPairDocument
from zbench.score import Score

def calculate_elos_sage(w, document_ids=None):
    """
    Engine swap: Call our SageMath high-precision Elo calculation service
    
    Args:
        w: numpy array preference matrix (n x n)
        document_ids: list of document IDs (optional, will use indices if not provided)
    
    Returns:
        elos: numpy array of Elo scores
        status: dict with convergence info
    """
    n = len(w)
    
    # Generate document IDs if not provided
    if document_ids is None:
        document_ids = [f"doc_{i}" for i in range(n)]
    
    # Convert preference matrix to sparse comparisons format
    comparisons = []
    for i in range(n):
        for j in range(n):
            if w[i, j] > 0 and i != j:  # Skip diagonal and zero entries
                comparisons.append({
                    "i": i,
                    "j": j,
                    "weight": float(w[i, j])
                })
    
    # Prepare input for SageMath service
    input_data = {
        "document_ids": document_ids,
        "comparisons": comparisons,
        "config": {
            "model": "bradley-terry",
            "epsilon": 1e-4,
            "max_iters": 1000
        }
    }
    
    # Use temporary files for communication
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
        json.dump(input_data, input_file)
        input_path = input_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        # Find the SageMath service script
        sage_script = Path(__file__).parent.parent.parent / "sage" / "elo_calculation_service.sage"
        
        if not sage_script.exists():
            print(f"Warning: SageMath service not found at {sage_script}")
            print("Falling back to internal calculate_elos (less precise)")
            from zbench.utils import calculate_elos
            return calculate_elos(w)
        
        # Call SageMath service
        print(f"  Calling SageMath high-precision engine...")
        result = subprocess.run(
            ["sage", str(sage_script), input_path, output_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Warning: SageMath service failed: {result.stderr}")
            print("Falling back to internal calculate_elos")
            from zbench.utils import calculate_elos
            return calculate_elos(w)
        
        # Read results
        with open(output_path, 'r') as f:
            output_data = json.load(f)
        
        # Convert scores back to numpy array in correct order
        elos = np.zeros(n)
        for i, doc_id in enumerate(document_ids):
            elos[i] = output_data['scores'][doc_id]
        
        # Print status
        status = output_data['status']
        print(f"  ✓ SageMath engine: Converged={status['converged']}, Iterations={status['iterations']}")
        
        return elos, status
        
    finally:
        # Clean up temp files
        os.unlink(input_path)
        os.unlink(output_path)


class EnsembleZELOAnnotatorSage:
    """Modified annotator using SageMath high-precision Elo calculation"""
    
    def __init__(self, dataset_path: str, annotated_dataset_path: str, *, cycle_num: int = 4, document_limit: int | None = None):
        self.initial_path = Path(dataset_path)
        self.dataset_name = self.initial_path.stem
        self.working_dir = Path(f"data/annotation/{self.dataset_name}")
        self.initial_dir = Path(self.initial_path).parent
        self.cycle_num = cycle_num
        self.document_limit = document_limit
        
        # File paths
        self.dataset_path = self.working_dir / "dataset.jsonl"
        self.pairs_path = self.working_dir / "pairs.json"
        self.ai_scores_path = self.working_dir / "ai_scores.json"
        self.annotated_dataset_path = annotated_dataset_path
        
        # Ensure folder exists
        self.working_dir.mkdir(parents=True, exist_ok=True)
        copy2(self.initial_path, self.dataset_path)
    
    async def step1_load_dataset(self) -> Dataset:
        """Load dataset from the folder."""
        print(f"Step 1: Loading dataset")
        
        dataset : Dataset = Dataset.model_validate({"data": load_jsonl(str(self.dataset_path))})
        print(f"Loaded {len(dataset.data)} queries from {self.dataset_path}")
        return dataset
    
    def get_random_cycle_pairs(self, n : int) -> list[tuple[int, int]]:
        """Get random cycle pairs from the dataset."""
        values = list(range(n))
        random.shuffle(values)
        pairs = []
        for i in range(len(values)):
            pairs.append((values[i], values[(i + 1) % n]))
        return pairs

    def step2_create_pairs(self, dataset: Dataset) -> DatasetPairs:
        """Create pairs from the dataset."""
        print("Step 2: Creating pairs for AI scoring")
        
        pairs : DatasetPairs = DatasetPairs(pairs=[])
        for data in tqdm(dataset.data, desc="Creating pairs"):
            query = data.query
            documents = data.documents
            n = len(data.documents)
            if self.document_limit is not None:
                n = min(n, self.document_limit)
            for _ in range(self.cycle_num):
                indices = self.get_random_cycle_pairs(n)
                for i, j in indices:
                    pairs.pairs.append(DatasetPair(
                        pair_id=f"{query.id}-{documents[i].id}-{documents[j].id}",
                        query_id=query.id,
                        query=query.query,
                        document_a=DatasetPairDocument(
                            document_id=documents[i].id,
                            metadata=documents[i].metadata,
                            content=documents[i].content,
                        ),
                        document_b=DatasetPairDocument(
                            document_id=documents[j].id,
                            metadata=documents[j].metadata,
                            content=documents[j].content,
                        ),
                    ))
            
        # Save pairs
        save_pydantic_json(self.pairs_path, pairs)
        print(f"Created {len(pairs.pairs)} pairs and saved to {self.pairs_path}")

        return pairs
    
    async def step3_score_pairs(self) -> None:
        """Score pairs."""
        print("Step 3: Scoring pairs")
        
        score = Score(pairs_path=str(self.pairs_path), scores_path=str(self.ai_scores_path))
        pairs = score.load_pairs()
        scores = await score.score_pairs(pairs)
        score.save_scores(scores)
        print(f"Scored {len(scores.scored_pairs)} pairs and saved to {self.ai_scores_path}")
    
    def step4_compose_annotated_dataset(self) -> None:
        """Compose annotated dataset using SageMath high-precision Elo calculation"""
        print("Step 4: Composing annotated dataset with SageMath engine")

        # Load AI scores
        ai_scores : DatasetPairScoredPairs = DatasetPairScoredPairs.model_validate(load_json(str(self.ai_scores_path)))
        dataset : Dataset = Dataset.model_validate({"data": load_jsonl(str(self.dataset_path))})

        annotated_dataset : AnnotatedDataset = AnnotatedDataset(data=[])

        scores : dict[str, DatasetPairScoredPairs] = defaultdict(lambda: DatasetPairScoredPairs(scored_pairs=[]))
        for pair in ai_scores.scored_pairs:
            scores[pair.pair.query_id].scored_pairs.append(pair)
        
        # Process each query
        for data in tqdm(dataset.data, desc="Calculating Elos with SageMath"):
            documents = data.documents
            id_to_idx = {doc.id: i for i, doc in enumerate(documents)}
            n = len(documents)
            w = np.zeros((n, n))
            for i in range(n):
                w[i,i] = 0.5
            for scored_pair in scores[data.query.id].scored_pairs:
                i = id_to_idx[scored_pair.pair.document_a.document_id]
                j = id_to_idx[scored_pair.pair.document_b.document_id]
                score = 0.0
                if scored_pair.openai_score.score > 0:
                    score += 1/3
                if scored_pair.anthropic_score.score > 0:
                    score += 1/3
                if scored_pair.gemini_score.score > 0:
                    score += 1/3
                w[j,i] += score
                w[i,j] += 1 - score
            
            # ENGINE SWAP: Use our high-precision SageMath calculation
            document_ids = [doc.id for doc in documents]
            elos, status = calculate_elos_sage(w, document_ids)
            
            annotated_dataset.data.append(AnnotatedQueryDocuments(
                query=data.query,
                documents=[AnnotatedDocument(
                    id=documents[i].id,
                    content=documents[i].content,
                    metadata=documents[i].metadata,
                    score=elos[i]
                ) for i in range(n)]
            ))
        
        save_pydantic_jsonl(self.annotated_dataset_path, annotated_dataset.data)

        print(f"Created {len(annotated_dataset.data)} annotated lines")
        print(f"Saved annotated dataset to {self.annotated_dataset_path}")
    
    async def zelo_annotate(self) -> None:
        """Run the complete pipeline with SageMath engine."""
    
        try:
            print("\n🚀 Starting ZBench pipeline with SageMath high-precision engine")
            
            # Step 1: Load data
            data = await self.step1_load_dataset()
            
            # Step 2: Create pairs
            pairs = self.step2_create_pairs(data)
            
            # Step 3: Score pairs
            await self.step3_score_pairs()

            # Step 4: Compose annotated dataset with SageMath engine
            self.step4_compose_annotated_dataset()
            
            print(f"\n🎉 Annotation completed successfully with high-precision Elos!")
            print(f"Final outputs:")
            print(f"Annotated dataset: {self.annotated_dataset_path}")
            
        except Exception as e:
            print(f"❌ Annotation failed: {e}")
            raise

async def main():
    parser = argparse.ArgumentParser(description="Annotate dataset with SageMath high-precision Elo calculation")
    parser.add_argument("dataset_path", help="Path to the dataset jsonl file")
    parser.add_argument("annotated_dataset_path", help="Path to the dataset jsonl file")
    parser.add_argument("--cycle_num", type=int, default=4, help="Number of cycles to run")
    parser.add_argument("--document_limit", type=int, default=None, help="Number of documents to use for scoring")
    args = parser.parse_args()
    
    processor = EnsembleZELOAnnotatorSage(
        dataset_path=args.dataset_path,
        annotated_dataset_path=args.annotated_dataset_path,
        cycle_num=args.cycle_num,
        document_limit=args.document_limit,
    )
    
    await processor.zelo_annotate()


if __name__ == "__main__":
    asyncio.run(main())