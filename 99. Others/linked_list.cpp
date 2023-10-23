#include <iostream>

// Node represents each element in the linked list
class Node {
    public:
        int data;       // Data of the node
        Node* next;     // Pointer to the next node

        // Constructor to initialize a node
        Node(int Value) : data(Value), next(nullptr) {}
};

// LinkedList class represents the linked list
class LinkedList {
    private:
        Node* head; // Pointer to the first node in the list
    
    public:
        // Constructor to initialize an empty linked list
        LinkedList() : head(nullptr) {}

        // Function to insert a new element at the end of the list
        void insert(int value) {
            Node* newNode = new Node(value);
            if (head == nullptr) {
                head = newNode;
            } else {
                Node* current = head;
                while (current->next != nullptr) {
                    current = current->next;
                }
                current->next = newNode;
            }
        }

        // Function to display the elements of the list
        void display() {
            Node* current = head;
            while (current != nullptr) {
                std::cout << current->data << " ";
                current = current->next;
            }
            std::cout << std::endl;
        }
};

int main() {
    // Creating a linked list and inserting elements
    LinkedList myList;

    myList.insert(1);
    myList.insert(2);
    myList.insert(3);

    // Displaying the elements of the linked list
    std::cout << "Linked List: ";
    myList.display();

    return 0;
}