import { FaGithub, FaLinkedin, FaEnvelope } from 'react-icons/fa';
import group from "/group.png";
import sir from "/sir.png"
import dhuruv from "/dhuruv.jpg"
import shruti from "/shruti.jpg"
import khushi from "/khushi.jpg"

const About = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header with Group Photo */}
      <header className="relative bg-blue-600 text-white">
        <div className="container mx-auto px-6 py-16 text-center">
          <h1 className="text-4xl font-bold mb-4">Meet the Analyza Team</h1>
          <p className="text-xl mb-8">The minds behind the Data Analysis & Visualization Tool</p>
          
          {/* Placeholder for group photo - replace with your actual image */}
          <div className="mx-auto w-full max-w-4xl bg-blue-400 rounded-lg shadow-xl overflow-hidden mb-8 flex items-center justify-center">
            <span className="text-white text-lg"> <img 
                src={group} 
                alt="group photo" 
                className="" 
              /></span>
          </div>
        </div>
      </header>

      {/* Team Members Section */}
      <section className="container mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* Shruti Srivastava */}
          <div className="bg-white rounded-xl shadow-md overflow-hidden transition-transform duration-300 hover:scale-105">
            <div className="h-48 bg-blue-100 flex items-center justify-center">
              {/* Placeholder for photo - replace with actual image */}
              <div className="w-32 h-32 rounded-full bg-blue-300 flex items-center justify-center">
                <span className="text-blue-600"><img 
                src={shruti} 
                alt="shruti" 
                className="rounded-full" 
              /></span>
              </div>
            </div>
            <div className="p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">Shruti Srivastava</h2>
              <p className="text-blue-600 font-medium mb-4">Full Stack Developer</p>
              <p className="text-gray-600 mb-4">
                Full-stack developer responsible for implementing the React.js frontend and FastAPI backend. 
                Designed the interactive UI components and integrated visualization libraries.
              </p>
              <div className="flex space-x-4">
                <a href="https://github.com/shru2811" className="text-gray-500 hover:text-blue-600">
                  <FaGithub size={20} />
                </a>
                <a href="https://www.linkedin.com/in/shru2003/" className="text-gray-500 hover:text-blue-600">
                  <FaLinkedin size={20} />
                </a>
                <a href="mailto:sshruti.2811@gmail.com" className="text-gray-500 hover:text-blue-600">
                  <FaEnvelope size={20} />
                </a>
              </div>
            </div>
          </div>

          {/* Dhuruv Kumar */}
          <div className="bg-white rounded-xl shadow-md overflow-hidden transition-transform duration-300 hover:scale-105">
            <div className="h-48 bg-blue-100 flex items-center justify-center">
              {/* Placeholder for photo - replace with actual image */}
              <div className="w-32 h-28 rounded-full bg-blue-300 flex items-center justify-center">
                <span className="text-blue-600"><img 
                src={dhuruv} 
                alt="Dhuruv" 
                className="rounded-full" 
              /></span>
              </div>
            </div>
            <div className="p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">Dhuruv Kumar</h2>
              <p className="text-blue-600 font-medium mb-4">Backend Developer & Project Visionary</p>
              <p className="text-gray-600 mb-4">
                Conceptualized the core idea of Analyza and implemented the backend architecture. 
                Developed the machine learning integration and API endpoints for data processing.
              </p>
              <div className="flex space-x-4">
                <a href="http://github.com/dhuruv3421" className="text-gray-500 hover:text-blue-600">
                  <FaGithub size={20} />
                </a>
                <a href="http://linkedin.com/in/dhuruv-kumar-16ba92291" className="text-gray-500 hover:text-blue-600">
                  <FaLinkedin size={20} />
                </a>
                <a href="mailto:dhuruvkumar2001@gmail.com" className="text-gray-500 hover:text-blue-600">
                  <FaEnvelope size={20} />
                </a>
              </div>
            </div>
          </div>

          {/* Khushi Chauhan */}
          <div className="bg-white rounded-xl shadow-md overflow-hidden transition-transform duration-300 hover:scale-105">
            <div className="h-48 bg-blue-100 flex items-center justify-center">
              {/* Placeholder for photo - replace with actual image */}
              <div className="w-32 h-32 rounded-full bg-blue-300 flex items-center justify-center">
                <span className="text-blue-600"><img 
                src={khushi} 
                alt="khushi" 
                className="rounded-full" 
              /></span>
              </div>
            </div>
            <div className="p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">Khushi Chauhan</h2>
              <p className="text-blue-600 font-medium mb-4">Research Engineer & Technical Author</p>
              <p className="text-gray-600 mb-4">
                Conducted research on data analysis techniques and contributed to backend development. 
                Primary author of project documentation including the SRS report and user guides.
              </p>
              <div className="flex space-x-4">
                <a href="https://github.com/Khushi20Chauhan" className="text-gray-500 hover:text-blue-600">
                  <FaGithub size={20} />
                </a>
                <a href="https://www.linkedin.com/in/khushi-chauhan-25a33a242/" className="text-gray-500 hover:text-blue-600">
                  <FaLinkedin size={20} />
                </a>
                <a href="mailto:Khushi.chauhann22@gmail.com" className="text-gray-500 hover:text-blue-600">
                  <FaEnvelope size={20} />
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Mentor Section */}
      <section className="bg-gray-100 py-12">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center text-gray-800 mb-12">Our Mentor</h2>
          
          <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-md overflow-hidden md:flex">
            <div className="md:w-1/3 bg-blue-100 flex items-center justify-center p-6">
              {/* Placeholder for mentor photo - replace with actual image */}
              <div className="w-40 h-40 rounded-full bg-blue-300 flex items-center justify-center">
                <span className="text-blue-600 text-center"><img 
                src={sir} 
                alt="Deepak Sir" 
                className="w-40 h-40 rounded-full" 
              /></span>
              </div>
            </div>
            <div className="p-8 md:w-2/3">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">Dr. Deepak Kumar Sharma</h2>
              <p className="text-blue-600 font-medium mb-4">Project Mentor</p>
              <p className="text-gray-600 mb-4">
                Assistant Professor (SG) System Cluster, School of Computer Science, UPES. 
                Guided the team through the entire project lifecycle with valuable technical 
                insights and research direction.
              </p>
              <div className="flex space-x-4">
                <a href="https://www.linkedin.com/in/deepak-sharma-phd-5bb45a1a/" className="text-gray-500 hover:text-blue-600">
                  <FaLinkedin size={20} />
                </a>
                <a href="mailto:dksharma@ddn.upes.ac.in" className="text-gray-500 hover:text-blue-600">
                  <FaEnvelope size={20} />
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Project Info */}
      <section className="container mx-auto px-6 py-12 text-center">
        
        <div className="bg-blue-50 p-6 rounded-lg inline-block">
          <p className="text-gray-700">
            <span className="font-semibold">Project Duration:</span> January 2025 - May 2025
          </p>
        </div>
      </section>
    </div>
  );
};

export default About;