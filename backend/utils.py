import numpy as np
from typing import List, Tuple, Any
from qdrant_client.models import ScoredPoint

def rank_list(search_result: List[ScoredPoint]) -> List[Tuple[int, int]]:
    """Convert search results into a ranked list of IDs with their positions.
    
    Args:
        search_result: List of ScoredPoint from Qdrant search
        
    Returns:
        List of (id, rank) tuples where rank starts from 1
    """
    return [(point.id, rank + 1) for rank, point in enumerate(search_result)]

def rrf(rank_lists, alpha=20, default_rank=1000):
    """
    Reciprocal Rank Fusion (RRF) using NumPy for large rank lists.
    
    RRF is a technique for combining multiple ranking algorithms' results
    into a single improved ranking. It gives higher weight to items that 
    appear near the top of multiple lists.
    
    Args:
        rank_lists: A list of rank lists. Each rank list is a list of (item, rank) tuples.
        alpha: The parameter alpha used in the RRF formula. Default is 20.
        default_rank: The default rank assigned to items not present in a rank list. Default is 1000.
        
    Returns:
        Sorted list of items based on their RRF scores.
    """
    # Consolidate all unique items from all rank lists
    all_items = set(item for rank_list in rank_lists for item, _ in rank_list)

    # Create a mapping of items to indices
    item_to_index = {item: idx for idx, item in enumerate(all_items)}

    # Initialize a matrix to hold the ranks, filled with the default rank
    rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)

    # Fill in the actual ranks from the rank lists
    for list_idx, rank_list in enumerate(rank_lists):
        for item, rank in rank_list:
            rank_matrix[item_to_index[item], list_idx] = rank

    # Calculate RRF scores using NumPy operations
    rrf_scores = np.sum(1.0 / (alpha + rank_matrix), axis=1)

    # Sort items based on RRF scores
    sorted_indices = np.argsort(-rrf_scores)  # Negative for descending order

    # Retrieve sorted items
    sorted_items = [(list(item_to_index.keys())[idx], rrf_scores[idx]) for idx in sorted_indices]

    return sorted_items

def format_search_results(results):
    """Format search results for display.
    
    Args:
        results: List of document results from Qdrant
        
    Returns:
        Formatted string for display
    """
    output = []
    for i, result in enumerate(results):
        output.append(f"{i+1}. ID: {result.id}")
        output.append(f"   {result.payload.get('text', 'No text available')}")
        output.append("")
    
    return "\n".join(output)