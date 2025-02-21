## Object-Oriented Programming (OOP) in Python

Object-Oriented Programming (OOP) is a paradigm that structures code using objects, which encapsulate data and behavior. Python supports OOP through **classes** and **instances**.

---

### **1. Classes and Instances**
A **class** is a blueprint for creating objects. An **instance** is an individual object created from a class.

#### **Example:**
```python
class Car:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year

# Creating instances (objects)
car1 = Car("Toyota", "Corolla", 2022)
car2 = Car("Honda", "Civic", 2023)

print(car1.brand)  # Output: Toyota
print(car2.model)  # Output: Civic
```

---

### **2. Instance Methods**
Instance methods are functions defined inside a class that operate on **instance attributes**.

#### **Example:**
```python
class Car:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year

    def display_info(self):
        return f"{self.brand} {self.model} ({self.year})"

car1 = Car("Ford", "Mustang", 2021)
print(car1.display_info())  # Output: Ford Mustang (2021)
```

---

### **3. Understanding `self`**
- `self` represents the instance of the class.
- It allows access to instance attributes and methods.
- Every method in a class must include `self` as the first parameter.

#### **Example:**
```python
class Dog:
    def __init__(self, name):
        self.name = name
    
    def bark(self):
        return f"{self.name} says Woof!"

dog1 = Dog("Buddy")
print(dog1.bark())  # Output: Buddy says Woof!
```

---

### **4. Object Initialization (`__init__` Method)**
- `__init__` is a special method (constructor) used for initializing objects.
- It automatically runs when an instance is created.

#### **Example:**
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person1 = Person("Alice", 30)
print(person1.name, person1.age)  # Output: Alice 30
```

---

### **5. Class vs. Instance Attributes**
- **Instance Attributes**: Unique to each instance.
- **Class Attributes**: Shared across all instances.

#### **Example:**
```python
class Circle:
    pi = 3.14159  # Class attribute

    def __init__(self, radius):
        self.radius = radius  # Instance attribute

circle1 = Circle(5)
circle2 = Circle(10)

print(circle1.pi)  # Output: 3.14159
print(circle1.radius)  # Output: 5
```

---

### **6. Inheritance**
Inheritance allows one class (child) to inherit attributes and methods from another (parent).

#### **Example:**
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def make_sound(self):
        return "Some sound"

class Dog(Animal):  # Dog inherits from Animal
    def make_sound(self):
        return "Bark!"

dog = Dog("Max")
print(dog.name)  # Output: Max
print(dog.make_sound())  # Output: Bark!
```

---

### **7. Encapsulation (Private and Public Attributes)**
- **Public attributes**: Accessible from outside the class.
- **Private attributes**: Prefix with `_` (protected) or `__` (private).

#### **Example:**
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        self.__balance += amount
    
    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())  # Output: 1500
# print(account.__balance)  # AttributeError: private variable
```

---

### **8. Polymorphism**
Polymorphism allows different classes to use the same method names with different implementations.

#### **Example:**
```python
class Bird:
    def sound(self):
        return "Chirp"

class Dog:
    def sound(self):
        return "Bark"

animals = [Bird(), Dog()]
for animal in animals:
    print(animal.sound())  
# Output: Chirp
# Output: Bark
```

---

### **9. OOP with `scikit-learn`**
Scikit-learn follows an OOP approach where models are objects.

#### **Example:**
```python
from sklearn.linear_model import LinearRegression

# Creating an instance of the model
model = LinearRegression()

# Checking methods available
print(dir(model))
```

---

### **10. Special Methods (`__str__`, `__repr__`, `__len__`)**
These methods define how objects behave in certain situations.

#### **Example:**
```python
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __str__(self):
        return f"'{self.title}' by {self.author}"

book = Book("1984", "George Orwell")
print(book)  # Output: '1984' by George Orwell
```

---

### **11. Magic Methods (Dunder Methods)**
Magic methods start and end with double underscores (`__`).

#### **Example:**
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(2, 3)
v2 = Vector(5, 6)
result = v1 + v2  # Uses __add__
print(result.x, result.y)  # Output: 7 9
```

---

### **12. Abstract Classes**
Abstract classes define methods that must be implemented by subclasses.

#### **Example:**
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side * self.side

square = Square(4)
print(square.area())  # Output: 16
```

---

### **13. Static and Class Methods**
- `@staticmethod`: Method that doesn't use `self`.
- `@classmethod`: Method that works with the class instead of an instance.

#### **Example:**
```python
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

print(MathUtils.add(5, 3))  # Output: 8
```

---

### **14. Properties with `@property` Decorator**
The `@property` decorator allows getter/setter-like functionality.

#### **Example:**
```python
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

person = Person("Alice")
print(person.name)  # Output: Alice
person.name = "Bob"
print(person.name)  # Output: Bob
```
---

### **15. Summary of Key Terms**
| Term | Definition |
|------|------------|
| **Class** | A blueprint for creating objects. |
| **Instance** | A specific object created from a class. |
| **Method** | A function defined inside a class. |
| **Instance Attribute** | A variable specific to an object. |
| **Class Attribute** | A variable shared across instances. |
| **Inheritance** | A class inherits from another class. |
| **Polymorphism** | Different classes use the same method names differently. |
| **Encapsulation** | Hiding data inside a class. |
| **Magic Methods** | Special methods with `__` (e.g., `__init__`, `__str__`). |

---

## Digging in Deeper into Abstract Super Class
### **Using `if`, `elif`, and `else` in an Abstract Superclass (OOP in Python)**

When working with **Object-Oriented Programming (OOP)** in Python, **abstract superclasses** define a blueprint for child classes. In such cases, `if`, `elif`, and `else` are used for **method logic, input validation, and handling different subclass behaviors.**  

---

### **1. What is an Abstract Superclass?**
- An **abstract superclass** is a class that **cannot be instantiated**.
- It defines **abstract methods** that **must** be implemented by subclasses.
- It is useful for enforcing method structure across different child classes.

✅ **In Python, `ABC` (Abstract Base Class) from the `abc` module is used to define abstract superclasses.**

---

### **2. Example: Abstract Payment System**
Let’s create an abstract superclass `Payment` with a `process_payment` method that is implemented differently in child classes like `CreditCardPayment` and `PayPalPayment`.

#### **Abstract Superclass (Payment)**
```python
from abc import ABC, abstractmethod

class Payment(ABC):
    def __init__(self, amount):
        self.amount = amount

    @abstractmethod
    def process_payment(self):
        """Abstract method to process payment."""
        pass
```

#### **Concrete Subclasses**
Each subclass implements the `process_payment` method, using `if-elif-else` for different payment conditions.

```python
class CreditCardPayment(Payment):
    def __init__(self, amount, card_type):
        super().__init__(amount)
        self.card_type = card_type

    def process_payment(self):
        if self.card_type == "Visa":
            print(f"Processing Visa payment of ${self.amount}")
        elif self.card_type == "MasterCard":
            print(f"Processing MasterCard payment of ${self.amount}")
        else:
            print("Invalid card type. Payment failed.")

class PayPalPayment(Payment):
    def __init__(self, amount, email):
        super().__init__(amount)
        self.email = email

    def process_payment(self):
        if "@" in self.email:
            print(f"Processing PayPal payment of ${self.amount} from {self.email}")
        else:
            print("Invalid PayPal email. Payment failed.")
```

#### **Using the Classes**
```python
payment1 = CreditCardPayment(100, "Visa")
payment1.process_payment()

payment2 = CreditCardPayment(150, "Discover")
payment2.process_payment()

payment3 = PayPalPayment(200, "user@example.com")
payment3.process_payment()
```
**Output:**
```
Processing Visa payment of $100
Invalid card type. Payment failed.
Processing PayPal payment of $200 from user@example.com
```

✅ **Here, `if-elif-else` is used inside the subclasses to handle different payment conditions.**  

---

### **3. Example: Abstract Shape Class (Handling Different Shapes)**
Another common use case for abstract classes is when different subclasses represent different types of objects.

#### **Abstract Superclass (Shape)**
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass
```

#### **Concrete Subclasses (`Circle` and `Rectangle`)**
```python
import math

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)
```

#### **Using the Classes**
```python
shape_type = "circle"

if shape_type == "circle":
    shape = Circle(5)
elif shape_type == "rectangle":
    shape = Rectangle(4, 6)
else:
    print("Invalid shape type")

print("Area:", shape.area())
print("Perimeter:", shape.perimeter())
```

**Output:**
```
Area: 78.53981633974483
Perimeter: 31.41592653589793
```

✅ **Here, `if-elif-else` is used to dynamically determine which shape class to instantiate.**

---

### **4. Example: Factory Pattern with Abstract Superclass**
Using `if-elif-else` inside a factory function helps create different instances dynamically.

```python
def get_payment_method(method, amount):
    if method == "credit":
        return CreditCardPayment(amount, "Visa")
    elif method == "paypal":
        return PayPalPayment(amount, "user@example.com")
    else:
        raise ValueError("Invalid payment method")

payment = get_payment_method("credit", 200)
payment.process_payment()
```
✅ **The `if-elif-else` inside `get_payment_method()` determines which class to instantiate dynamically.**

---

### **Key Takeaways**
| Concept | Explanation |
|---------|-------------|
| `ABC` Class | Used to create abstract superclasses that **enforce method implementation** in subclasses. |
| `if` in Abstract Class | **Not common**, but can be used inside a method if logic depends on a condition. |
| `if-elif-else` in Subclasses | Used to **handle different conditions dynamically** (e.g., different card types, payment methods). |
| Factory Pattern | Uses `if-elif-else` to **dynamically create subclass instances** based on input. |

## Mixins and Multiple Inheritance
### **Advanced OOP Concepts: Mixins and Multiple Inheritance**  

When working with **Object-Oriented Programming (OOP)** in Python, **mixins** and **multiple inheritance** provide more flexibility and code reuse.  

---

## **1. Multiple Inheritance**  
Multiple inheritance allows a class to inherit from more than one parent class.  
🔹 **Why use it?** When a class needs to combine functionality from multiple sources.  
🔹 **Risk?** It can cause **ambiguity** in method resolution.  

### **Example: Multiple Inheritance with Animal Classes**
```python
class Walker:
    def walk(self):
        return "I can walk."

class Swimmer:
    def swim(self):
        return "I can swim."

class Duck(Walker, Swimmer):  # Inherits from both Walker and Swimmer
    def sound(self):
        return "Quack!"
```

#### **Using the Duck Class**
```python
donald = Duck()
print(donald.walk())  # I can walk.
print(donald.swim())  # I can swim.
print(donald.sound())  # Quack!
```
✅ **Duck inherits both `walk()` from `Walker` and `swim()` from `Swimmer`.**  

---

## **2. The Diamond Problem and MRO (Method Resolution Order)**  
🔹 **What is the Diamond Problem?**  
If multiple classes inherit from the same parent, method resolution can become ambiguous.  

### **Example: Diamond Problem**
```python
class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B")

class C(A):
    def show(self):
        print("C")

class D(B, C):  # Inherits from both B and C
    pass
```

#### **Method Resolution Order (MRO)**
```python
obj = D()
obj.show()  # Output: B
print(D.mro())  # [D, B, C, A, object]
```
🔹 **Python uses the C3 Linearization (MRO) to resolve method conflicts.**  
🔹 **Order:** `D -> B -> C -> A`.  
🔹 **First-come, first-served** (B is first, so `show()` from B is used).  

---

## **3. Mixins (A Lightweight Multiple Inheritance Approach)**
🔹 **What is a Mixin?**  
A **mixin** is a class that **only provides additional behavior** but is **not meant to be instantiated alone**.  

🔹 **Why use Mixins?**  
- To **add functionality** to multiple unrelated classes.  
- To **avoid deep inheritance trees**.  
- To **prevent code duplication**.  

### **Example: Logging Mixin**
```python
class LoggingMixin:
    def log(self, message):
        print(f"[LOG]: {message}")

class FileHandler(LoggingMixin):
    def save(self, filename):
        self.log(f"Saving file: {filename}")
        print(f"File {filename} saved.")

class DatabaseHandler(LoggingMixin):
    def connect(self, db_name):
        self.log(f"Connecting to database: {db_name}")
        print(f"Connected to {db_name}.")
```

#### **Using the Mixins**
```python
file_handler = FileHandler()
file_handler.save("report.pdf")

db_handler = DatabaseHandler()
db_handler.connect("users_db")
```
**Output:**
```
[LOG]: Saving file: report.pdf
File report.pdf saved.
[LOG]: Connecting to database: users_db
Connected to users_db.
```
✅ **Both `FileHandler` and `DatabaseHandler` inherit the `log()` method from `LoggingMixin` without unnecessary code duplication.**  

---

## **4. Mixins with Multiple Inheritance**
You can **combine mixins with multiple inheritance** for greater flexibility.  

### **Example: Combining Authentication and Logging Mixins**
```python
class LoggingMixin:
    def log(self, message):
        print(f"[LOG]: {message}")

class AuthenticationMixin:
    def authenticate(self, user):
        if user == "admin":
            print("Authentication successful.")
        else:
            print("Access denied.")

class SecureFileHandler(LoggingMixin, AuthenticationMixin):
    def delete(self, filename, user):
        self.authenticate(user)
        self.log(f"Deleting file: {filename}")
        print(f"File {filename} deleted.")
```

#### **Using SecureFileHandler**
```python
handler = SecureFileHandler()
handler.delete("data.csv", "admin")  # Authentication successful, file deleted
handler.delete("data.csv", "guest")  # Access denied, no deletion
```
✅ **Mixins allow us to mix and match functionalities without deep inheritance chains.**  

---

## **5. Abstract Base Classes (ABC) + Mixins**
Mixins can work **alongside abstract classes** for more control.  

### **Example: Abstract Base Class + Mixin**
```python
from abc import ABC, abstractmethod

class NotifiableMixin:
    def notify(self, message):
        print(f"Notification: {message}")

class Payment(ABC):
    @abstractmethod
    def process_payment(self):
        pass

class CreditCardPayment(Payment, NotifiableMixin):
    def process_payment(self):
        self.notify("Processing Credit Card Payment")
        print("Credit card payment processed.")

class PayPalPayment(Payment, NotifiableMixin):
    def process_payment(self):
        self.notify("Processing PayPal Payment")
        print("PayPal payment processed.")
```

#### **Using the Classes**
```python
payment = CreditCardPayment()
payment.process_payment()

paypal = PayPalPayment()
paypal.process_payment()
```
✅ **Abstract base classes enforce structure, while mixins provide additional functionality like notifications.**  

---

## **Key Takeaways**
| Concept | Explanation |
|---------|-------------|
| **Multiple Inheritance** | A class inherits from more than one parent. |
| **Method Resolution Order (MRO)** | Determines which method is called first in multiple inheritance. |
| **Diamond Problem** | When a method is inherited from multiple parents, causing ambiguity. |
| **Mixins** | Lightweight classes that provide extra functionality without being instantiated. |
| **Mixins + Abstract Classes** | Abstract classes enforce structure, while mixins add reusable functionality. |

---

### **Should You Use Multiple Inheritance?**
🔹 **YES** if:
- You are combining **independent features** (e.g., Logging + Authentication).
- You need **reusable behaviors** across different classes.

🔹 **NO** if:
- It leads to **complexity and ambiguity**.
- You can **use composition instead**.

---
