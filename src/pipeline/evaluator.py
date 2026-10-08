from typing import List, Dict


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    """
    Calculate Levenshtein distance between two sequences using dynamic programming.
    Time: O(m*n), Space: O(m*n)
    """
    m, n = len(ref), len(hyp)
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Deletion cost
    for j in range(n + 1):
        dp[0][j] = j  # Insertion cost
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = min(
                    dp[i-1][j],      # Deletion
                    dp[i][j-1],      # Insertion
                    dp[i-1][j-1]     # Substitution
                ) + 1
    
    return dp[m][n]


class Evaluator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.all_refs = []
        self.all_hyps = []
        self.metrics = {}
    
    def add_sample(self, reference: str, hypothesis: str):
        """Add a sample for evaluation"""
        self.all_refs.append(reference)
        self.all_hyps.append(hypothesis)
    
    def calculate_wer(self) -> float:
        """Word Error Rate using custom implementation"""
        if not self.all_refs:
            return 0.0
        
        total_errors = 0
        total_words = 0
        
        for ref, hyp in zip(self.all_refs, self.all_hyps):
            ref_words = ref.split()
            hyp_words = hyp.split()
            
            if len(ref_words) == 0:
                continue
            
            # Calculate edit distance at word level
            errors = edit_distance(ref_words, hyp_words)
            
            total_errors += errors
            total_words += len(ref_words)
        
        return total_errors / total_words if total_words > 0 else 0.0
    
    def calculate_cer(self) -> float:
        """Character Error Rate using custom implementation"""
        if not self.all_refs:
            return 0.0
        
        total_errors = 0
        total_chars = 0
        
        for ref, hyp in zip(self.all_refs, self.all_hyps):
            # Remove spaces for character-level comparison
            ref_chars = list(ref.replace(" ", ""))
            hyp_chars = list(hyp.replace(" ", ""))
            
            if len(ref_chars) == 0:
                continue
            
            # Calculate edit distance at character level
            errors = edit_distance(ref_chars, hyp_chars)
            
            total_errors += errors
            total_chars += len(ref_chars)
        
        return total_errors / total_chars if total_chars > 0 else 0.0
    
    def calculate_wrr(self) -> float:
        """Word Recognition Rate = 1 - WER"""
        return 1 - self.calculate_wer()
    
    def get_detailed_report(self) -> Dict:
        """Get full evaluation report"""
        self.metrics = {
            "wer": self.calculate_wer(),
            "cer": self.calculate_cer(),
            "wrr": self.calculate_wrr(),
            "total_samples": len(self.all_refs),
        }
        return self.metrics
    
    def print_report(self):
        """Print evaluation results"""
        metrics = self.get_detailed_report()
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Word Error Rate (WER): {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
        print(f"Character Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
        print(f"Word Recognition Rate (WRR): {metrics['wrr']:.4f} ({metrics['wrr']*100:.2f}%)")
        print("="*50)
    
    def get_sample_comparisons(self, n: int = 5) -> List[Dict]:
        """Get side-by-side comparisons"""
        comparisons = []
        for i in range(min(n, len(self.all_refs))):
            comparisons.append({
                "reference": self.all_refs[i],
                "hypothesis": self.all_hyps[i],
                "sample_id": i
            })
        return comparisons