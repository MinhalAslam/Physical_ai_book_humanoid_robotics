import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

function BookDescription() {
  return (
    <div className="container">
      <div className="row">
        <div className="col col--12">
          <div className="text--center padding-vert--md">
            <Heading as="h2" className={styles.mainTitle}>
              Physical AI & Humanoid Robotics book
            </Heading>
            <p className={styles.bookDescription}>
              This book covers the principles and practical foundations of Physical AI and Humanoid Robotics, focusing on intelligent systems that interact with the real world. It explains how artificial intelligence is integrated with sensors, actuators, control systems, and robotic bodies to create embodied intelligence. The book explores perception, motion planning, learning, and human–robot interaction, with real-world examples and simulations. Designed for students and beginners, it bridges theory and practice to prepare readers for building and understanding intelligent humanoid systems.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <BookDescription />
    </section>
  );
}
