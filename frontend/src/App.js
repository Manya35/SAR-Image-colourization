import React, { useState, useEffect } from "react";
import { Link } from "react-scroll";
import { FaLinkedin, FaInstagram, FaGlobe } from "react-icons/fa";
import "./App.css";

function App() {
  const [showNavbar, setShowNavbar] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      const heroHeight = window.innerHeight;
      setShowNavbar(scrollY > heroHeight - 100); 
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="app">
      {/* Fullscreen Title Section */}
      <section id="home" className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">SAR IMAGE COLOURIZATION</h1>
          <p className="hero-subtitle">
            Transforming Monochrome Radar Data ​into Vibrant Insights
          </p>
          <button className="cta-button">Try It Now!</button>
        </div>
      </section>

      {/* Navbar */}
      {showNavbar && (
        <nav className="navbar">
          <ul className="navbar-links">
            <li>
              <Link to="home" smooth={true} className="navbar-item">
                SAR IMAGE COLOURIZATION
              </Link>
            </li>
            <li>
              <Link to="working" smooth={true} className="navbar-item">
                How It Works?
              </Link>
            </li>
            <li>
              <Link to="colourise" smooth={true} className="navbar-item">
                Colourise My Image
              </Link>
            </li>
            <li>
              <Link to="footer" smooth={true} className="navbar-item">
                About Us / Contact
              </Link>
            </li>
          </ul>
        </nav>
      )}

      {/* Working Section */}
      <section id="working" className="working-section">
        <h2 className="section-title">HOW IT WORKS?</h2>
        <div className="timeline">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className="timeline-card">
              <h3>Step {i + 1}</h3>
              <p>Description of the process step {i + 1}.</p>
            </div>
          ))}
        </div>
      </section>

      {/* Colourise Section */}
      <section id="colourise" className="colourise-section">
        <h2 className="section-title">COLOUR THE SAR IMAGE</h2>
        <div className="colourise-content">
          <div className="colourise-animation">
            <p>Animation Placeholder</p>
          </div>
          <div className="upload-section">
            <input type="file" className="upload-input" />
            <button className="cta-button">Colourise</button>
          </div>
        </div>
      </section>

      {/* Information Section */}
      <section id="information" className="information-section">
        <h2 className="section-title">Information</h2>
        <div className="information-content">
          <div className="sar-image">
            <h3>SAR Image</h3>
            <p>Placeholder for SAR Image</p>
          </div>
          <div className="colourised-image">
            <h3>Colourised Image</h3>
            <p>Placeholder for Colourised Image</p>
          </div>
          <div className="evaluation-metrics">
            <h3>Evaluation Matrices</h3>
            <p>PSNR: 42.1 dB</p>
            <p>SSIM: 0.98</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer id="footer" className="footer">
        <div className="footer-columns">
          <div className="footer-column">
            <h3>Project Paper</h3>
            <a href="https://example.com" target="_blank" rel="noopener noreferrer">
              Review Paper Link
            </a>
          </div>
          {[
            { name: "Manya", role: "Frontend Developer", linkedin: "#", instagram: "#", website: "#" },
            { name: "Palak", role: "Backend Developer", linkedin: "#", instagram: "#", website: "#" },
            { name: "Shanvi", role: "Data Scientist", linkedin: "#", instagram: "#", website: "#" },
          ].map((member, index) => (
            <div key={index} className="footer-column">
              <h3>{member.name}</h3>
              <p>{member.role}</p>
              <div className="social-icons">
                <a href={member.linkedin} target="_blank" rel="noopener noreferrer">
                  <FaLinkedin />
                </a>
                <a href={member.instagram} target="_blank" rel="noopener noreferrer">
                  <FaInstagram />
                </a>
                <a href={member.website} target="_blank" rel="noopener noreferrer">
                  <FaGlobe />
                </a>
              </div>
              <p>2nd Year, IGDTUW</p>
            </div>
          ))}
        </div>
        <p className="footer-note">© 2024 SAR Image Colourization Project</p>
      </footer>
    </div>
  );
}

export default App;
