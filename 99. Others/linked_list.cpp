#include <iostream>

// Node represents each element in the linked list
class Node {
    public:
        int data; // Data of the node
        Node* next; // Pointer to the next node

    // Constructor to initialize a node    
    Node(int value) {
        data = value;
        next = nullptr;
    }
};

// LinkedList class represents the linked list
class LinkedList {
    private: 
        Node* head; // Pointer to the first node in the list

    public:
        // Constructor to initialize an empty linked list
        LinkedList() {
            head = nullptr;
        }

        // Function to insert a new element at the end of the list
        void insert(int value) {
            Node* newNode = new Node(value);

            // If the list is empty, the new node becomes the head.
            if (head == nullptr) {
                head = newNode;
            }
            // If the list is not empty, the function traverses the list until it finds the last node, and then adds the new node to the end 
            else {
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
            while(current != nullptr) {
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
};

// pointer concept that makes the insertions and deletions in the middle process more efficient compared to array (without pointer concept)
// Unlike arrays, linked lists do not require pre-allocation of memory for a fixed size
// but pointer can lead to overhead especially when dealing with large data due to its storage
// pointer will gain its full advantage while there is frequent data changes 
