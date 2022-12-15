# Leetcode DSA in Java

https://neetcode.io/practice

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

Implementation using stack:
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
    public ListNode removeNthFromEnd(ListNode head, int n) {
        Stack<ListNode> stack = new Stack<>();

        ListNode currentNode = head;
        while(currentNode != null){
            stack.push(currentNode);
            currentNode = currentNode.next;
        }

        // Boundary cases
        if(n == stack.size()){
            return head.next;
        }

        if(n == 1){
            stack.pop();
            ListNode beforeNthNode = stack.peek();
            beforeNthNode.next = null;
            return head;
        }

        for(int i = 0; i < n - 2; i++){
            stack.pop();
        }
        
        ListNode afterNthNode = stack.pop();
        ListNode nthNode = stack.pop();
        ListNode beforeNthNode = stack.pop();

        beforeNthNode.next = afterNthNode;

        return head;
    }
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
    List<List<Integer>> solList = new LinkedList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // Arrays.sort(candidates);
        generateCombs(candidates, target, 0, new LinkedList<Integer>(), 0, 0);
        return this.solList;
    }

    private void generateCombs(
        int[] candidates, 
        int target, 
        int sum, 
        LinkedList<Integer> sol, 
        int index,
        int numCombs
    ){
        if(numCombs >= 150 || sum > target){
            return;
        }
        
        if(sum == target){
            this.solList.add(new LinkedList<>(sol));
        }

        for(int i = index; i < candidates.length; i++){
            sol.push(candidates[i]);
            generateCombs(candidates, target, sum + candidates[i], sol, i, numCombs + 1);
            sol.pop();
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
    List<List<Integer>> solList = new LinkedList<>();

    public List<List<Integer>> permute(int[] nums) {
        generatePerms(nums, new HashSet<Integer>(), new LinkedList<>());
        return this.solList;
    }

    private void generatePerms(int[] nums, HashSet<Integer> set, LinkedList<Integer> sol){
        if(set.size() == nums.length){
            this.solList.add(new LinkedList<Integer>(sol));
            return;
        }
        for(int i = 0; i < nums.length; i++){
            if(set.contains(nums[i])) continue; // Candidate considered
            set.add(nums[i]);
            sol.push(nums[i]);
            generatePerms(nums, set, sol);
            set.remove(nums[i]);
            sol.pop();
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
    List<List<Integer>> solList = new LinkedList<>();

    public List<List<Integer>> subsets(int[] nums) {
        generateSubsets(nums, 0, new LinkedList<Integer>());
        return this.solList;
    }

    private void generateSubsets(int[] nums, int start, LinkedList<Integer> sol) {
        this.solList.add(new LinkedList<>(sol));

        for(int i = start; i < nums.length; i++){
            sol.push(nums[i]);
            generateSubsets(nums, i + 1, sol);
            sol.pop();
        }
    }
}
```

### 12. Subsets of Number Array (With Duplicates) - https://leetcode.com/problems/subsets-ii/description/
```java
class Solution {
    List<List<Integer>> solList = new LinkedList<>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        generateSubsets(nums, 0, new LinkedList<Integer>());
        return this.solList;
    }

    private void generateSubsets(int[] nums, int start, LinkedList<Integer> sol){
        this.solList.add(new LinkedList<>(sol));

        for(int i = start; i < nums.length; i++){
            if(i > start && nums[i - 1] == nums[i]) continue;
            sol.push(nums[i]);
            generateSubsets(nums, i + 1, sol);
            sol.pop();
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

### 20. Longest Substring Without Repeating Characters - https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        // Sliding window approach
        int start = 0;
        int end = 0;
        HashSet<Character> window = new HashSet<>();
        int maxLength = 0;
        
        while(end < s.length()){
            if(window.contains(s.charAt(end))){
                // window shrinking
                window.remove(s.charAt(start++));
            } else {
                // window growing
                window.add(s.charAt(end++));
                maxLength = Math.max(window.size(), maxLength);
                System.out.println(maxLength);
            }
        }

        return maxLength;
    }
}
```

### 21. Group Anagrams - https://leetcode.com/problems/group-anagrams/description/

#### Using Arrays.sort():
```java
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.Arrays;

class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {      
        Map<String, List<String>> solMap = new HashMap<>(strs.length);
        
        for(int i = 0; i < strs.length; i++){            
            String key = generateKey(strs[i]);
            if(solMap.containsKey(key)){
                solMap.get(key).add(strs[i]);
            } else {
                List<String> groupedList = new ArrayList<>();
                groupedList.add(strs[i]);
                solMap.put(key, groupedList);
            }
        }
        return new ArrayList<>(solMap.values());
    }
    
    public String generateKey(String string){
        if(string.length() == 0){ return string; }
        
        char[] charArray = string.toCharArray();
        Arrays.sort(charArray);
        return String.valueOf(charArray);
    }
}
```

#### Without using Arrays.sort():
```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, LinkedList<String>> map = new HashMap<>();

        for(int i = 0; i < strs.length; i++){
            int[] chars = new int[26];
            for(int j = 0; j < strs[i].length(); j++){
                chars[(int)(strs[i].charAt(j) - 'a')]++;
            }
            StringBuilder sb = new StringBuilder();
            for(int k = 0; k < chars.length; k++){
                sb.append(String.valueOf(chars[k]));
                sb.append(",");
            }
            String key = sb.toString();

            if(map.containsKey(key)){
                map.get(key).add(strs[i]);
            } else {
                LinkedList<String> newList = new LinkedList<>();
                newList.add(strs[i]);
                map.put(key, newList);
            }
        }

        List<List<String>> solList = new LinkedList<>();
        map.forEach((key, value) -> {
            solList.add(value);
        });

        return solList;
    }
}
```

### 22. Longest Repeating Character Replacement - https://leetcode.com/problems/longest-repeating-character-replacement/
Could'nt Solve!

### 23. 3Sum - https://leetcode.com/problems/3sum/description/
```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        return solve(nums, 0);
    }

    private List<List<Integer>> solve(int[] nums, int target){
        HashSet<List<Integer>> set = new HashSet<>();
        Arrays.sort(nums);

        for(int i = 0; i < nums.length - 2; i++){
            int j = i + 1;
            int k = nums.length - 1;

            while(j < k){
                int sum = nums[i] + nums[j] + nums[k];

                if(sum == target){
                    List<Integer> sol = new LinkedList<>();
                    sol.add(nums[i]);
                    sol.add(nums[j++]);
                    sol.add(nums[k--]);
                    set.add(sol);
                    continue;
                }

                if(sum < target){
                    // We are less than target, array is sorted, we need to increase sum
                    j++;
                    continue;
                }

                if(sum > target){
                    // We are more than target, array is sorted, we need to reduce sum
                    k--;
                }
            }
        }

        return new ArrayList<>(set);
    }
}
```

### 24. Number of Islands - https://leetcode.com/problems/number-of-islands/description/

* Find land starting location
* Increment numIslands
* Then, recursively mark all the touching land locations by assigning '-'
* Continue traversing the grid
```java
class Solution {
    int solution = 0;

    public int numIslands(char[][] grid) {
        for(int row = 0; row < grid.length; row++){
            for(int col = 0; col < grid[0].length; col++){
                if(grid[row][col] == '1'){
                    // Found the starting of an island
                    this.solution++;
                    // Mark the territory as explored
                    this.exploreIsland(grid, row, col);
                }
            }
        }
        return this.solution;
    }

    private void exploreIsland(char[][] grid, int row, int col){
        if(
            row < 0 ||
            col < 0 ||
            row > grid.length - 1 ||
            col > grid[0].length - 1
        ){
            // Out of bounds
            return;
        }
        if(grid[row][col] != '1'){
            // This is not a land, return
            return;
        }
        grid[row][col] = '-'; // Visited
        
        exploreIsland(grid, row - 1, col); // top
        exploreIsland(grid, row + 1, col); // bottom
        exploreIsland(grid, row, col - 1); // left
        exploreIsland(grid, row, col + 1); // top
    }
}
```

### 25. Kth Smallest Element in BST - https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/

* Concept - Remember inorder traversal of a BST is always sorted!
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
    List<Integer> traversalList = new LinkedList<>();

    public int kthSmallest(TreeNode root, int k) {
        this.inorderTraversal(root);
        
        int counter = 0;
        ListIterator<Integer> iterator = this.traversalList.listIterator();
        while (iterator.hasNext()){
            if(counter == k - 1){
                break;
            }
            counter++;
            iterator.next();
        }
        return iterator.next();
    }

    private void inorderTraversal(TreeNode node){
        if(node == null){
            return;
        }

        // Left, Node, Right
        inorderTraversal(node.left);
        this.traversalList.add(node.val);
        inorderTraversal(node.right);
    }
}
```

### 26. - Construct BST with Preorder and Inorder Traversal - https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/

You need to remember this question. I am unable to understand it.
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
    int iterations = 0;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < inorder.length; i++){
            map.put(inorder[i], i);
        }
        return createNodes(0, preorder.length - 1, preorder, inorder, map);
    }

    private TreeNode createNodes(int preStart, int preEnd, int[] preorder, int[] inorder, HashMap<Integer, Integer> map){
        if(preStart > preEnd || this.iterations > preorder.length - 1){
            return null;
        }

        int indexInorder = map.get(preorder[this.iterations]);
        TreeNode node = new TreeNode(preorder[this.iterations]);

        this.iterations++;

        node.left = createNodes(preStart, indexInorder - 1, preorder, inorder, map);
        node.right = createNodes(indexInorder + 1, preEnd, preorder, inorder, map);

        return node;
    }
}
```

### 27. Kth Largest Element - https://leetcode.com/problems/kth-largest-element-in-an-array/description/

* Simply use a heap and poll elements
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(nums.length, Comparator.reverseOrder());

        for(int i = 0; i < nums.length; i++){
            maxHeap.add(nums[i]);
        }

        for(int j = 0; j < k - 1; j++){
            maxHeap.poll(); // Remember poll is faster than remove if you want to remove the root element everytime
        }

        return maxHeap.peek();
    }
}
```

### 28. Check Sudoku - https://leetcode.com/problems/valid-sudoku/description/

* Check row uniqueness with a set
* Check col uniqueness with a set
* Check region uniqueness with a set
```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        for(int row = 0; row < board.length; row++){
            if(!this.checkRow(board, row)) return false;
        }

        for(int col = 0; col < board[0].length; col++){
            if(!this.checkCol(board, col)) return false;
        }

        return (
            this.checkSection(board, 0, 2, 0, 2) &&
            this.checkSection(board, 3, 5, 0, 2) &&
            this.checkSection(board, 6, 8, 0, 2) &&
            this.checkSection(board, 0, 2, 3, 5) &&
            this.checkSection(board, 3, 5, 3, 5) &&
            this.checkSection(board, 6, 8, 3, 5) &&
            this.checkSection(board, 0, 2, 6, 8) &&
            this.checkSection(board, 3, 5, 6, 8) &&
            this.checkSection(board, 6, 8, 6, 8)
        );
    }

    private boolean checkRow(char[][] board, int row){
        HashSet<Character> set = new HashSet<>();
        for(int i = 0; i < board[row].length; i++){
            if(!checkChar(board[row][i])){
                continue;
            }
            if(set.contains(board[row][i])){
                // Repeated
                return false;
            }
            set.add(board[row][i]);
        }
        return true;
    }

    private boolean checkCol(char[][] board, int col){
        HashSet<Character> set = new HashSet<>();
        for(int i = 0; i < board.length; i++){
            if(!checkChar(board[i][col])){
                continue;
            }
            if(set.contains(board[i][col])){
                // Repeated
                return false;
            }
            set.add(board[i][col]);
        }
        return true;
    }

    private boolean checkSection(char[][] board, int rowStart, int rowEnd, int colStart, int colEnd){
        HashSet<Character> set = new HashSet<>();
        for(int i = rowStart; i <= rowEnd; i++){
            for(int j = colStart; j <= colEnd; j++){
                if(!checkChar(board[i][j])){
                    continue;
                }
                if(set.contains(board[i][j])){
                    // Repeated
                    return false;
                }
                set.add(board[i][j]);
                }
        }
        return true;
    }

    private boolean checkChar(char c){
        if(c >= '1' && c <= '9'){
            return true;
        }
        return false;
    }
}
```

### 29. Product of Array Except Self - https://leetcode.com/problems/product-of-array-except-self/description/

One of those questions which you have to remember.
* Populate res array with product of all elements before position i during pass from left to right
* On another pass from right to left, multiply the value in res at position i with product of all elements after i
```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];

        // Pre-Product
        int preProduct = 1;
        for(int i = 0; i < nums.length; i++){
            res[i] = preProduct;
            preProduct *= nums[i];
        }

        // Post-Product
        int postProduct = 1;
        for(int j = nums.length - 1; j >= 0; j--){
            res[j] *= postProduct;
            postProduct *= nums[j];
        }

        return res;
    }
}
```

### 30. Longest Consecutive Sequence - https://leetcode.com/problems/longest-consecutive-sequence/description/
```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if(nums.length == 0) return 0;

        HashMap<Integer, Integer> map = new HashMap<>();
        int max = 1;

        for(int i = 0; i < nums.length; i++){
            map.put(nums[i], i);
        }

        for(int i = 0; i < nums.length; i++){
            max = Math.max(findLongest(nums[i], map, nums), max);
        }

        return max;
    }

    private int findLongest(int num, HashMap<Integer, Integer> map, int[] nums){
        int current = num;
        int sol = 0;
        // Search forward
        while(map.containsKey(current)){
            nums[map.get(current)] = Integer.MIN_VALUE;
            sol++;
            current++;
            if(sol == map.size()) return sol;
        }
        // Search backward
        current = num - 1;
        while(map.containsKey(current)){
            nums[map.get(current)] = Integer.MIN_VALUE;
            sol++;
            current--;
            if(sol == map.size()) return sol;
        }
        return sol;
    }
}
```
