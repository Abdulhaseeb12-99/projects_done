public class Employee {
    private int id;
    private String firstName;
    private String middleName;
    private String lastName;
    private String position;
    private String salary;
    private double age;
    private String gender;
    private String phno;
    private String addr;
    private String email; // Changed postalCode to email

    public Employee(int id, String firstName, String middleName, String lastName, String position, String salary, double age, String gender, String phno, String addr, String email) { // Changed postalCode to email
        this.setId(id);
        this.setFirstName(firstName);
        this.setMiddleName(middleName);
        this.setLastName(lastName);
        this.setPosition(position);
        this.setSalary(salary);
        this.setAge(age);
        this.setGender(gender);
        this.setPhno(phno);
        this.setAddr(addr);
        this.setEmail(email); // Changed postalCode to email
    }



	// Getters and setters
    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getMiddleName() {
        return middleName;
    }

    public void setMiddleName(String middleName) {
        this.middleName = middleName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getPosition() {
        return position;
    }

    public void setPosition(String position) {
        this.position = position;
    }


    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public String getPhno() {
        return phno;
    }

    public void setPhno(String phno) {
        this.phno = phno;
    }

    public String getAddr() {
        return addr;
    }

    public void setAddr(String addr) {
        this.addr = addr;
    }

    public String getEmail() { // Changed postalCode to email
        return email;
    }

    public void setEmail(String email) { // Changed postalCode to email
        this.email = email;
    }

	public double getAge() {
		return age;
	}

	public void setAge(double age) {
		this.age = age;
	}

	public String getSalary() {
		return salary;
	}

	public void setSalary(String salary) {
		this.salary = salary;
	}
}