# Leetcode DSA in Java

## MEDIUM

### 1. Delete Nth last node - https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/

You got to remember this only since this has a simple concept but confusing implementation with difficult to handle edge cases.
```java
public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode start = new ListNode(0);
    ListNode slow = start;
    ListNode fast = start;
    start.next = head;

    for(int i = 0; i <= n; i++ ) {
        fast = fast.next;
    }

    while(fast != null) {
        fast = fast.next;
        slow = slow.next;
    }

    // Delete the node
    slow.next = slow.next.next;

    return start.next;
}
```

### 2. Fruit Into Baskets - https://leetcode.com/problems/fruit-into-baskets/description/ 

My own solution:
```java
class Solution {
    public int totalFruit(int[] fruits) {
        return collectFruitsAroundATree(0, fruits, 0);
    }

    private int collectFruitsAroundATree(int index, int[] fruits, int maxCollected) {
        if(index > fruits.length - 1) return maxCollected;

        int fruitBasket1Type = -1;
        int fruitBasket2Type = -1;
        int currentCollected = 0;

        int i = index;

        // look back first
        while(i >= 0){
            if(fruitBasket1Type == -1 || fruitBasket1Type == fruits[i]){
                fruitBasket1Type = fruits[i--];
                currentCollected++;
            } else if(fruitBasket2Type == -1 || fruitBasket2Type == fruits[i]) {
                fruitBasket2Type = fruits[i--];
                currentCollected++;
            } else {
                // fruit cannot go in any basket!
                break;
            }
        }

        int j = index + 1;

        // look ahead later
        while(j < fruits.length){
            if(fruitBasket1Type == -1 || fruitBasket1Type == fruits[j]){
                fruitBasket1Type = fruits[j++];
                currentCollected++;
            } else if(fruitBasket2Type == -1 || fruitBasket2Type == fruits[j]) {
                fruitBasket2Type = fruits[j++];
                currentCollected++;
            } else {
                // fruit cannot go in any basket!
                break;
            }
        }

        return collectFruitsAroundATree(j, fruits, Math.max(maxCollected, currentCollected));
    }
}
```

### 3. Container with most water - https://leetcode.com/problems/container-with-most-water/description/ 

2 pointers:
```java
class Solution {
    public int maxArea(int[] height) {
        int leftPointer = 0;
        int rightPointer = height.length - 1;
        int maxVolume = 0;

        while(leftPointer < rightPointer) {
            int currentVolume = Math.min(height[leftPointer], height[rightPointer]) * 
            (rightPointer - leftPointer);
            maxVolume = Math.max(maxVolume, currentVolume);
            
            if(height[leftPointer] < height[rightPointer]){
                leftPointer++;
            } else {
                rightPointer--;
            }
        }

        return maxVolume;
    }
}
```

### 4. Implement a Min Stack - https://leetcode.com/problems/min-stack/description/

Stack:
```java
class MinStack {
    LinkedList<Integer> list = new LinkedList<Integer>();
    LinkedList<Integer> minList = new LinkedList<Integer>();
    int size = 0;
    
    public void push(int val) {
        list.push(val);
        size++;
        // we will track the minimum value for each push operation
        minList.push(Math.min(minList.size() == 0 ? Integer.MAX_VALUE : minList.peek(), val));
    }
    
    public void pop() {
        if(size == 0) return;

        list.pop();
        size--;

        minList.pop();
    }
    
    public int top() {
        return list.peek();
    }
    
    public int getMin() {
        return minList.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(val);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```

### 5. Word Search - https://leetcode.com/problems/word-search

Backtracking:
```java
class Solution {
    public boolean exist(char[][] board, String word) {
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                if(explore(board, word, i, j, 0)){
                    return true;   
                }
            }
        }
        
        return false;
    }
    
    private boolean explore(char[][] board, String word, int row, int col, int index){
        if(
            row < 0 ||
            row > board.length - 1 ||
            col < 0 ||
            col > board[0].length - 1 ||
            index > word.length() - 1
        ) {
            // out of bounds
            return false;
        }
        
        if(board[row][col] == word.charAt(index)) {
            // check if this is the last index
            if(index == word.length() - 1){
                return true;
            }
            
            char val = board[row][col];
            board[row][col] = '-';
            
            // Explore all directions since we have a match on this node and
            // return true if any direction matches
            if(
                explore(board, word, row - 1, col, index + 1) || // up
                explore(board, word, row + 1, col, index + 1) || // down
                explore(board, word, row, col - 1, index + 1) || // left
                explore(board, word, row, col + 1, index + 1) // right
            ) {
                return true;
            } else {
                board[row][col] = val;
            }
        }
        
        return false;
    }
}
```

### 6. Rotten Oranges - https://leetcode.com/problems/rotting-oranges/description/
```java
class Solution {
    public int orangesRotting(int[][] grid) {
        int timePassed = -1;
        int freshOranges = 0;
        Queue<int[]> rottenOranges = new LinkedList<int[]>();

        for(int row = 0; row < grid.length; row++){
            for(int col = 0; col < grid[0].length; col++){
                if(grid[row][col] == 2){
                    rottenOranges.add(new int[]{row, col});
                } else if (grid[row][col] == 1) {
                    freshOranges++;
                }
            }
        }

        if(freshOranges == 0) return 0; // All rotten
        if(rottenOranges.size() == 0) return -1; // All fresh

        while(rottenOranges.size() != 0) {
            timePassed++;
            int size = rottenOranges.size();

            for(int i = 0; i < size; i++){
                int[] currentOrange = rottenOranges.remove();
                int row = currentOrange[0];
                int col = currentOrange[1];

                // All surrounding will rot, add to queue
                if(row - 1 >= 0 && grid[row - 1][col] == 1){
                    rottenOranges.add(new int[]{row - 1, col});
                    grid[row - 1][col] = 2; // rotten
                    freshOranges--;
                }
                if(row + 1 < grid.length && grid[row + 1][col] == 1){
                    rottenOranges.add(new int[]{row + 1, col});
                    grid[row + 1][col] = 2; // rotten
                    freshOranges--;
                }
                if(col - 1 >= 0 && grid[row][col - 1] == 1){
                    rottenOranges.add(new int[]{row, col - 1});
                    grid[row][col - 1] = 2; // rotten
                    freshOranges--;
                }
                if(col + 1 < grid[0].length && grid[row][col + 1] == 1){
                    rottenOranges.add(new int[]{row, col + 1});
                    grid[row][col + 1] = 2; // rotten
                    freshOranges--;
                }
            }
        }

        return freshOranges > 0 ? -1 : timePassed;
    }
}
```

### 7. Combination Sum (Can Reuse Element) - https://leetcode.com/problems/combination-sum/description/
```java
class Solution {
    private List<List<Integer>> resultList = new LinkedList<List<Integer>>();
    
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        generateCombinations(0, 0, target, 0, new LinkedList<Integer>(), candidates);

        return resultList;
    }

    private void generateCombinations(
        int numCombs,
        int totalSum,
        int target,
        int start,
        LinkedList<Integer> combs,
        int[] candidates
    ){
        if(numCombs >= 150 || totalSum > target) return;
        if(totalSum == target) this.resultList.add(new LinkedList<Integer>(combs));

        // At any stage, all candidates should be considered, repeat candidates are allowed
        for(int i = start; i < candidates.length; i++) {
            combs.add(candidates[i]);

            generateCombinations(
                numCombs + 1,
                totalSum + candidates[i],
                target,
                i, // not i + 1 since we are also considering duplicates
                combs,
                candidates
            );

            combs.removeLast();
        }
    }
}
```

### 8. Combination Sum (Can’t Reuse Element) - https://leetcode.com/problems/combination-sum-ii/description/
```java
class Solution {
    private List<List<Integer>> resultList = new LinkedList<List<Integer>>();

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        generateCombs(
            candidates,
            target,
            0,
            0,
            new LinkedList<Integer>()
        );
        return this.resultList;
    }

    private void generateCombs(
        int[] candidates,
        int target,
        int currentSum,
        int start,
        LinkedList<Integer> tempList
    ){
        if(currentSum > target) return; // invalid situation
        if(currentSum == target) this.resultList.add(new LinkedList<Integer>(tempList));

        for(int i = start; i < candidates.length; i++) {
            if(i > start && candidates[i] == candidates[i - 1]) continue; // No duplicates

            tempList.add(candidates[i]);

            generateCombs(
                candidates,
                target,
                currentSum + candidates[i],
                i + 1,
                tempList
            );

            tempList.removeLast();
        }
    }
}
```

### 9. Permutations of Number Array (Without Duplicates) - https://leetcode.com/problems/permutations/description/
```java
class Solution {
    List<List<Integer>> resultList = new LinkedList<List<Integer>>();

    public List<List<Integer>> permute(int[] nums) {
        generatePerms(nums, new LinkedList<Integer>());

        return this.resultList;
    }

    private void generatePerms(int[] nums, LinkedList<Integer> tempList){
        if(tempList.size() == nums.length){
            this.resultList.add(new LinkedList<Integer>(tempList));
            return;
        }

        // At any position we have to consider all values that are left except the values
        // chosen already

        for(int i = 0; i < nums.length; i++) {
            if(tempList.contains(nums[i])) continue; // We have already chosen this element

            tempList.add(nums[i]);
            generatePerms(nums, tempList);
            tempList.removeLast(); // Backtrack cleanup step
        }
    }
}
```

### 10. Permutations of Number Array (With Duplicates) - https://leetcode.com/problems/permutations-ii/description/
```java
class Solution {
    List<List<Integer>> resultList = new LinkedList<List<Integer>>();

    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        generatePerms(nums, new LinkedList<Integer>(), new boolean[nums.length]);
        return resultList;
    }

    private void generatePerms(
        int[] nums,
        LinkedList<Integer> result,
        boolean[] used
        ) {
        if(result.size() == nums.length){
            this.resultList.add(new LinkedList(result));
            return; // we are done now, backtrack
        }

        for(int i = 0; i < nums.length; i++){
            if(used[i] || (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])) continue;

            result.add(nums[i]);
            used[i] = true;
            generatePerms(nums, result, used);

            // backtrack cleanup step
            result.removeLast();
            used[i] = false;
        }
    }
}
```

### 11. Subsets of Number Array (Without Duplicates) - https://leetcode.com/problems/subsets/description/
```java
class Solution {
    private List<List<Integer>> resultList = new LinkedList<List<Integer>>();

    public List<List<Integer>> subsets(int[] nums) {
        Arrays.sort(nums);
        generateSubsets(nums, new LinkedList<Integer>(), 0);
        return this.resultList;
    }

    private void generateSubsets(int[] nums, LinkedList<Integer> result, int start) {
        if(result.size() > nums.length) {
            return;
        }
        
        this.resultList.add(new LinkedList<Integer>(result));

        for(int i = start; i < nums.length; i++) {
            result.add(nums[i]);
            generateSubsets(nums, result, i + 1);
            result.removeLast();
        }
    }
}
```

### 12. Subsets of Number Array (With Duplicates) - https://leetcode.com/problems/subsets-ii/description/
```java
class Solution {
    List<List<Integer>> resultList = new LinkedList<List<Integer>>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        generateSubsets(nums, new LinkedList<Integer>(), 0);
        return this.resultList;
    }

    private void generateSubsets(
        int[] nums,
        LinkedList<Integer> result,
        int start
    ) {
        this.resultList.add(new LinkedList<Integer>(result));

        for(int i = start; i < nums.length; i++) {
            // This is to prevent duplicates
            if(i > start && nums[i] == nums[i - 1]) continue; 

            result.add(nums[i]);
            generateSubsets(nums, result, i + 1);

            // Backtrack cleanup step
            result.removeLast();
        }
    }
}
```

### 13. Palindrome Partitioning - https://leetcode.com/problems/palindrome-partitioning/description/ 
```java
class Solution {
    List<List<String>> resultList = new LinkedList<List<String>>();

    public List<List<String>> partition(String s) {
        generatePartitions(s, new LinkedList<String>(), 0);

        return this.resultList;
    }

    private void generatePartitions(
        String s,
        LinkedList<String> result,
        int start
    ) {
        if(start == s.length()){
            this.resultList.add(new LinkedList<String>(result));
        } else {
            for(int i = start; i < s.length(); i++) {
                if(this.checkPalindrome(s, start, i)) {
                    result.add(s.substring(start, i + 1));
                    generatePartitions(s, result, i + 1);

                    // backtrack cleanup
                    result.removeLast();
                }
            }
        }
    }

    private boolean checkPalindrome(String s, int low, int high) {
        while(low < high){
            if(s.charAt(low++) != s.charAt(high--)) return false;
        }
        return true;
    }
}
```

### 14. Binary Tree Level Order Traversal - https://leetcode.com/problems/binary-tree-level-order-traversal/description/
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    List<List<Integer>> solList = new ArrayList<List<Integer>>();

    public List<List<Integer>> levelOrder(TreeNode root) {
        if(root == null) return this.solList;

        traverse(root, 1);
        return this.solList;
    }

    public void traverse(TreeNode node, int level){
        if(this.solList.size() < level){
            // this level has not been reached yet
            this.solList.add(new LinkedList<Integer>());
        }
        this.solList.get(level - 1).add(node.val);

        if(node.left != null) traverse(node.left, level + 1);
        if(node.right != null) traverse(node.right, level + 1);
    }
}
```

### 15. Validate Binary Search Tree - https://leetcode.com/problems/validate-binary-search-tree/description/
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public boolean isValidBST(TreeNode root) {
        return validateAtNode(root);
    }

    private boolean validateAtNode(TreeNode node){
        if(node != null){
            if(
                checkLeftNodes(node.left, node.val) &&
                checkRightNodes(node.right, node.val)
            ){
                // Tree is valid till now, go deeper
                return validateAtNode(node.left) && validateAtNode(node.right);
            } else {
                return false;
            }
        }
        return true;
    }

    private boolean checkLeftNodes(TreeNode node, int val){
        if(node != null){
            return node.val < val && 
            checkLeftNodes(node.left, val) && 
            checkLeftNodes(node.right, val);
        }
        return true;
    }

    private boolean checkRightNodes(TreeNode node, int val){
        if(node != null){
            return node.val > val && 
            checkRightNodes(node.left, val) && 
            checkRightNodes(node.right, val);
        }
        return true;
    }
}
```

### 16. Course Schedule - https://leetcode.com/problems/course-schedule/description/
Couldn't Solve!

### 17. Open the Lock - https://leetcode.com/problems/open-the-lock/description/
Couldn't Solve!

### 18. Reorder LinkedList - https://leetcode.com/problems/reorder-list/description/
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public void reorderList(ListNode head) {
        ArrayList<ListNode> list = new ArrayList<ListNode>();

        ListNode currentNode = head;
        while(currentNode.next != null){
            list.add(currentNode);
            currentNode = currentNode.next;
        }
        list.add(currentNode);

        int startIndex = 0;
        int endIndex = list.size() - 1;
        while(startIndex < endIndex){
            ListNode startNode = list.get(startIndex);
            ListNode endNode = list.get(endIndex);
            startNode.next = endNode;
            startIndex++;
            endIndex--;
            if(startIndex < endIndex){
                endNode.next = list.get(startIndex);
            } else if(startIndex == endIndex) {
                endNode.next = list.get(startIndex);
                endNode.next.next = null;
            } else {
                endNode.next = null;
            }
        }
    }
}
```

### 19. Top K Frequent Elements - https://leetcode.com/problems/top-k-frequent-elements/description/
```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> frequencyMap = new HashMap<>();
        int maxFrequency = 0;

        for(int i = 0; i < nums.length; i++){
            if(frequencyMap.containsKey(nums[i])){
                int frequency = frequencyMap.get(nums[i]);
                frequencyMap.put(nums[i], frequency + 1);
                maxFrequency = Math.max(frequency + 1, maxFrequency);
            } else {
                frequencyMap.put(nums[i], 1);
                maxFrequency = Math.max(1, maxFrequency);
            }
        }

        List<LinkedList<Integer>> frequencyBuckets = 
        new ArrayList<LinkedList<Integer>>(maxFrequency);

       for(int i = 0; i < maxFrequency; i++){
           frequencyBuckets.add(new LinkedList<Integer>());
       }

        frequencyMap.forEach((num, frequency) -> {
            frequencyBuckets.get(frequency - 1).add(num);
        });

        // return top k
        int[] sol = new int[k];
        int added = 0;
        int bucketIndex = frequencyBuckets.size() - 1;

        while(added < k && bucketIndex >= 0){
            ListIterator<Integer> iterator = frequencyBuckets.get(bucketIndex).listIterator();

            while(iterator.hasNext() && added < k){
                sol[added] = iterator.next();
                added++;
            }
            bucketIndex--;
        }

        return sol;
    }
}
```
