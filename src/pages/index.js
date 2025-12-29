import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className="hero-section">
      <div className="hero-container">
        <div className="hero-text-content">
          <h1 className="hero-title">
            Physical AI & Humanoid Robotics
          </h1>
          <p className="hero-subtitle">
            Your roadmap to mastering embodied intelligence — from ROS2 foundations to Digital Twins, Motion, Vision,
            and LLM-powered control systems.
          </p>
          <Link
            className="hero-button"
            to="/docs/intro">
            Start Reading
          </Link>
        </div>
        <div className="hero-illustration">
          <div className="robot-illustration">
            <img
              src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300' width='400' height='300'%3E%3Cdefs%3E%3ClinearGradient id='grad1' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%235d9bff;stop-opacity:0.3' /%3E%3Cstop offset='100%25' style='stop-color:%231a1a24;stop-opacity:0.3' /%3E%3C/linearGradient%3E%3C/defs%3E%3Crect x='50' y='20' width='300' height='260' rx='20' fill='url(%23grad1)' stroke='rgba(93, 155, 255, 0.3)' stroke-width='2'/%3E%3Ccircle cx='200' cy='80' r='40' fill='rgba(93, 155, 255, 0.2)' stroke='rgba(93, 155, 255, 0.4)' stroke-width='2'/%3E%3Ccircle cx='180' cy='70' r='5' fill='rgba(255, 255, 255, 0.8)'/%3E%3Ccircle cx='220' cy='70' r='5' fill='rgba(255, 255, 255, 0.8)'/%3E%3Crect x='170' y='120' width='60' height='100' rx='10' fill='rgba(93, 155, 255, 0.15)' stroke='rgba(93, 155, 255, 0.3)' stroke-width='2'/%3E%3Crect x='130' y='140' width='40' height='60' rx='5' fill='rgba(93, 155, 255, 0.1)' stroke='rgba(93, 155, 255, 0.2)' stroke-width='1'/%3E%3Crect x='230' y='140' width='40' height='60' rx='5' fill='rgba(93, 155, 255, 0.1)' stroke='rgba(93, 155, 255, 0.2)' stroke-width='1'/%3E%3Crect x='180' y='220' width='20' height='50' rx='5' fill='rgba(93, 155, 255, 0.1)' stroke='rgba(93, 155, 255, 0.2)' stroke-width='1'/%3E%3Crect x='200' y='220' width='20' height='50' rx='5' fill='rgba(93, 155, 255, 0.1)' stroke='rgba(93, 155, 255, 0.2)' stroke-width='1'/%3E%3Ccircle cx='200' cy='80' r='25' fill='none' stroke='rgba(93, 155, 255, 0.2)' stroke-width='1' stroke-dasharray='5,5'/%3E%3C/svg%3E"
              alt="Humanoid Robot"
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                borderRadius: '20px'
              }}
            />
          </div>
        </div>
      </div>
      <div className="decorative-circle circle-1"></div>
      <div className="decorative-circle circle-2"></div>
      <div className="decorative-circle circle-3"></div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
