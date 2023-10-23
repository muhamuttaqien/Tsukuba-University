#include <iostream>
#include <stack>

int main() {
    std::stack<int> myStack;

    myStack.push(1);
    myStack.push(2);
    myStack.push(3);

    std::cout << "Top of the Stack: " << myStack.top() << std::endl;

    myStack.pop();

    std::cout << "Top of the Stack after pop: " << myStack.top() << std::endl;

    return 0;
}