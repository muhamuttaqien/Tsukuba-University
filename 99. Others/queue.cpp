#include <iostream>
#include <queue>

int main() {
    std::queue<int> myQueue;

    myQueue.push(1);
    myQueue.push(2);
    myQueue.push(3);

    std::cout << "Front of the Queue: " << myQueue.front() << std::endl;

    myQueue.pop();

    std::cout << "Front of the Queue after pop: " << myQueue.front() << std::endl;

    return 0;
}